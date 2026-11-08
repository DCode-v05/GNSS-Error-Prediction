import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional
import json

from src.models.model import EnsembleModel
from src.data.data_processor import GNSSDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Trainer class for the GNSS error prediction model.
    """
    def __init__(self, config: Dict):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = EnsembleModel(config).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate']
        )
        self.criterion = nn.MSELoss()
        
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    batch_size: int) -> DataLoader:
        """
        Prepare data for training.
        
        Args:
            X: Input features
            y: Target values
            batch_size: Batch size
            
        Returns:
            DataLoader for the dataset
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            self.optimizer.zero_grad()
            loss.backward()
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
        total_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              patience: int = 10) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        train_loader = self.prepare_data(
            X_train, y_train,
            batch_size=self.config['batch_size']
        )
        val_loader = self.prepare_data(
            X_val, y_val,
            batch_size=self.config['batch_size']
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'ensemble_weights': []
        }
        
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            weights = self.model.get_model_weights().cpu().detach().numpy()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['ensemble_weights'].append(weights.tolist())
            
            logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"LSTM Weight: {weights[0]:.3f}, Transformer Weight: {weights[1]:.3f}"
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
                
        return history
        
    def save_model(self):
        """
        Save the model and training configuration.
        """
        model_path = self.save_dir / 'model.pth'
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        config_path = self.save_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        logger.info(f"Config saved to {config_path}")
        
    def load_model(self):
        """
        Load a saved model.
        """
        model_path = self.save_dir / 'model.pth'
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path))
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"No saved model found at {model_path}")

def main():
    """
    Main training function.
    """
    config = {
        'input_size': 5,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100,
        'save_dir': 'models/saved/20250930_203944',
        'data_path': 'data/raw/DATA_GEO_Train.csv',
        'sequence_length': 10
    }
    
    data_processor = GNSSDataProcessor(save_dir=Path(config['save_dir']))
    data = data_processor.prepare_data(data_path=config['data_path'], sequence_length=config['sequence_length'])
    X_train, X_val, y_train, y_val = data['X_train'], data['X_val'], data['y_train'], data['y_val']
    config['input_size'] = X_train.shape[-1]
    
    trainer = ModelTrainer(config)
    
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    history_path = Path(config['save_dir']) / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    logger.info(f"Training history saved to {history_path}")

if __name__ == '__main__':
    main()