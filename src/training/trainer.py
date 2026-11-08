import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

from src.models.advanced_model import MultiHorizonTransformer, GaussianNLLLoss

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config: Dict):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = MultiHorizonTransformer(config['model']).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-5)
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config['training'].get('lr_patience', 5),
            verbose=True
        )
        
        self.criterion = GaussianNLLLoss()
        self.best_val_loss = float('inf')
        self.patience = config['training'].get('patience', 10)
        self.early_stop_counter = 0
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        self.save_dir = Path(config['training'].get('save_dir', 'models/saved'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series data.
        
        Args:
            X: Input features [samples, features]
            y: Target values [samples, targets]
            seq_length: Length of each sequence
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        Xs, ys = [], []
        
        for i in range(len(X) - seq_length):
            Xs.append(X[i:(i + seq_length)])
            ys.append(y[i + seq_length - 1])
            
        return np.array(Xs), np.array(ys)
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, batch_size: int, 
                    val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for training and validation.
        
        Args:
            X: Input features [samples, features]
            y: Target values [samples, targets]
            batch_size: Batch size
            val_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        seq_length = self.config['data']['sequence_length']
        X_seq, y_seq = self.create_sequences(X, y, seq_length)
        
        split_idx = int(len(X_seq) * (1 - val_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_X)
            
            loss = 0.0
            for horizon, pred in outputs.items():
                horizon_min = int(horizon) // 15
                target = batch_y[:, horizon_min-1:horizon_min]
                loss += self.criterion(
                    pred['mean'], 
                    target,
                    pred['std']
                )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                
                for horizon, pred in outputs.items():
                    horizon_min = int(horizon) // 15
                    target = batch_y[:, horizon_min-1:horizon_min]
                    loss = self.criterion(
                        pred['mean'],
                        target,
                        pred['std']
                    )
                    total_loss += loss.item()
        
        return total_loss / (len(val_loader) * len(outputs))
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """
        Train the model with cross-validation.
        
        Args:
            X: Input features [samples, features]
            y: Target values [samples, targets]
            
        Returns:
            Training history
        """
        train_loader, val_loader = self.prepare_data(
            X, y,
            batch_size=self.config['training']['batch_size'],
            val_split=0.2
        )
        
        for epoch in range(self.config['training']['epochs']):
            start_time = time.time()
            
            train_loss = self.train_epoch(train_loader)
            
            val_loss = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                self.save_checkpoint()
            else:
                self.early_stop_counter += 1
            
            logger.info(
                f"Epoch {epoch + 1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} - "
                f"Time: {time.time() - start_time:.2f}s"
            )
            
            if self.early_stop_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        self.save_checkpoint(f'model_final.pt')
        
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, filename: str = 'best_model.pt') -> None:
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.save_dir / filename
        torch.save({
            'epoch': len(self.history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.history['train_loss'][-1],
            'val_loss': self.history['val_loss'][-1],
            'config': self.config
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def save_history(self) -> None:
        """Save training history to disk."""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f)
        
        self.plot_training_curves()
    
    def plot_training_curves(self) -> None:
        """Plot training and validation loss curves."""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['lr'], label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.title('Learning Rate Schedule')
        plt.legend()
        
        plt.tight_layout()
        plot_path = self.save_dir / 'training_curves.png'
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved training curves to {plot_path}")
