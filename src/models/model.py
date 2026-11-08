from pathlib import Path
from typing import Dict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.output_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        return lstm_out

class TransformerModel(nn.Module):
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output_size = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class EnsembleModel(nn.Module):
    @staticmethod
    def load_model(model_dir: Path):
        with open(model_dir / 'config.json', 'r') as f:
            config = json.load(f)
        model = EnsembleModel(config)
        model.load_state_dict(torch.load(model_dir / 'best_model.pth'), strict=False)
        return model

    def __init__(self, config: Dict):
        super(EnsembleModel, self).__init__()
        input_size = config['input_size']
        lstm_hidden_size = config.get('lstm_hidden_size', 64)
        lstm_num_layers = config.get('lstm_num_layers', 2)
        transformer_d_model = config.get('transformer_d_model', 64)
        transformer_nhead = config.get('transformer_nhead', 8)
        transformer_num_layers = config.get('transformer_num_layers', 2)
        dropout = config.get('dropout', 0.1)
        
        self.lstm_input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        self.lstm_output_norm = nn.LayerNorm(lstm_hidden_size)
        
        self.transformer_input_norm = nn.LayerNorm(input_size)
        self.input_projection = nn.Linear(input_size, transformer_d_model)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=transformer_num_layers
        )
        self.transformer_output_norm = nn.LayerNorm(transformer_d_model)
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size + transformer_d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        combined_size = lstm_hidden_size + transformer_d_model
        self.output_layer = nn.Sequential(
            nn.Linear(combined_size, combined_size),
            nn.LayerNorm(combined_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_size, combined_size // 2),
            nn.LayerNorm(combined_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_size // 2, 1)
        )
        
        self._init_weights()
        
        self.ensemble_weights = nn.Parameter(torch.ones(2))
    
    def get_model_weights(self):
        return torch.softmax(self.ensemble_weights, dim=0)
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform for better gradient flow."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outlier_cols = []
        for i in range(x.shape[-1]):
            if f'error_is_outlier' in str(i):
                outlier_cols.append(i)
        
        if outlier_cols:
            features = x[..., [i for i in range(x.shape[-1]) if i not in outlier_cols]]
            outliers = x[..., outlier_cols]
        else:
            features = x
            outliers = None
        
        lstm_input = self.lstm_input_norm(features)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = self.lstm_output_norm(lstm_out)
        lstm_last = lstm_out[:, -1, :]
        
        transformer_input = self.transformer_input_norm(features)
        x_projected = self.input_projection(transformer_input)
        transformer_out = self.transformer(x_projected)
        transformer_out = self.transformer_output_norm(transformer_out)
        transformer_last = transformer_out[:, -1, :]
        
        combined = torch.cat([lstm_last, transformer_last], dim=1)
        
        combined_seq = combined.unsqueeze(1)
        attended_output, _ = self.temporal_attention(
            combined_seq, combined_seq, combined_seq
        )
        combined = attended_output.squeeze(1)
        
        output = self.output_layer(combined)
        if outliers is not None:
            outlier_weight = torch.sigmoid(outliers.mean(dim=1, keepdim=True))
            output = output * (1 + outlier_weight)
        
        return output.squeeze(-1)