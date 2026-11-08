import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime
from src.models.model import EnsembleModel
from src.data.data_processor import GNSSDataProcessor
from src.evaluation.metrics import calculate_metrics, evaluate_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

def select_features(X: np.ndarray, y: np.ndarray, k: int = 24) -> np.ndarray:
    """Select most important features using f_regression.
    
    Args:
        X: Input sequences of shape (n_samples, sequence_length, n_features)
        y: Target values of shape (n_samples,) or (n_samples, n_targets)
        k: Number of features to select
        
    Returns:
        Selected feature indices
    """
    X_last = X[:, -1, :]
    
    y_2d = y.reshape(-1, 1) if len(y.shape) == 1 else y
    
    selected_features = set()
    for i in range(y_2d.shape[1]):
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X_last, y_2d[:, i])
        selected_features.update(np.where(selector.get_support())[0])
    
    selected_features = sorted(list(selected_features))
    logger.info(f"Selected {len(selected_features)} features")
    
    return np.array(selected_features)

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                optimizer: optim.Optimizer, criterion: nn.Module,
                device: torch.device, max_epochs: int = 50,
                patience: int = 10, max_grad_norm: float = 0.5,
                warmup_epochs: int = 5, min_lr: float = 1e-6) -> dict:
    """Train model with early stopping, gradient clipping, and comprehensive metrics."""
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [], 'lr': [],
        'train_metrics': [], 'val_metrics': []
    }
    
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=max_epochs,
        min_lr=min_lr
    )
    
    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0
        batch_count = 0
        all_train_outputs = []
        all_train_targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            total_train_loss += loss.item()
            batch_count += 1
            
            all_train_outputs.append(output.detach())
            all_train_targets.append(target.detach())
            
            if batch_idx % 3 == 0:
                logger.info(f"Epoch {epoch}: [{batch_idx}/{len(train_loader)}] "
                          f"Train Loss: {loss.item():.6f}")
        
        avg_train_loss = total_train_loss / batch_count
        
        train_outputs = torch.cat(all_train_outputs).cpu()
        train_targets = torch.cat(all_train_targets).cpu()
        train_metrics = calculate_metrics(train_targets, train_outputs)
        
        model.eval()
        with torch.no_grad():
            all_val_outputs = []
            all_val_targets = []
            total_val_loss = 0
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_val_loss += loss.item()
                all_val_outputs.append(output)
                all_val_targets.append(target)
            
            val_outputs = torch.cat(all_val_outputs).cpu()
            val_targets = torch.cat(all_val_targets).cpu()
            val_metrics = calculate_metrics(val_targets, val_outputs)
            avg_val_loss = total_val_loss / len(val_loader)
        
        current_lr = scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        logger.info(
            f"Epoch {epoch}:\n"
            f"Train - Loss: {avg_train_loss:.6f}, MAE: {train_metrics['mae']:.6f}, "
            f"RMSE: {train_metrics['rmse']:.6f}, MAPE: {train_metrics['mape']:.2f}%\n"
            f"Val - Loss: {avg_val_loss:.6f}, MAE: {val_metrics['mae']:.6f}, "
            f"RMSE: {val_metrics['rmse']:.6f}, MAPE: {val_metrics['mape']:.2f}%\n"
            f"LR: {current_lr:.6f}"
        )
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {
                'state_dict': model.state_dict(),
                'metrics': val_metrics,
                'epoch': epoch
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
    
    history['best_model_state'] = best_model_state
    return history

def main():
    print("Main function started.")
    config = {
        'data_path': 'data/raw/DATA_GEO_Train.csv',
        'sequence_length': 24,
        'batch_size': 32,
        'lstm_hidden_size': 8,
        'lstm_num_layers': 1,
        'transformer_d_model': 8,
        'transformer_nhead': 2,
        'transformer_num_layers': 1,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'weight_decay': 0.1,
        'max_epochs': 50,
        'patience': 10,
        'max_grad_norm': 0.5,
        'n_splits': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    data_processor = GNSSDataProcessor()
    
    data_dict = data_processor.prepare_data(
        data_path=config['data_path'],
        sequence_length=config['sequence_length']
    )
    
    if data_dict is None:
        logger.error("Failed to prepare data")
        return
    
    print("Data processor initialized and data prepared.")
    
    X = np.concatenate([data_dict['X_train'], data_dict['X_val']], axis=0)
    y = np.concatenate([data_dict['y_train'], data_dict['y_val']], axis=0)
    
    selected_features = select_features(X, y)
    X = X[:, :, selected_features]
    
    print("Configuration, data processor, data loading, preprocessing, and feature selection initialized.")
    
    kfold = KFold(n_splits=config['n_splits'], shuffle=True, random_state=42)
    fold_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        logger.info(f"\nTraining fold {fold + 1}/{config['n_splits']}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
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
            batch_size=config['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size']
        )
        
        input_size = X_train.shape[2]
        model = EnsembleModel(
            input_size=input_size,
            lstm_hidden_size=config['lstm_hidden_size'],
            lstm_num_layers=config['lstm_num_layers'],
            transformer_d_model=config['transformer_d_model'],
            transformer_nhead=config['transformer_nhead'],
            transformer_num_layers=config['transformer_num_layers'],
            dropout=config['dropout']
        ).to(config['device'])
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        criterion = nn.MSELoss()
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=config['device'],
            max_epochs=config['max_epochs'],
            patience=config['patience'],
            max_grad_norm=config['max_grad_norm'],
            warmup_epochs=max(3, config['max_epochs'] // 10),
            min_lr=1e-6
        )
        
        fold_save_path = f'models/fold_{fold + 1}'
        os.makedirs(fold_save_path, exist_ok=True)
        
        torch.save(history['best_model_state']['state_dict'],
                  os.path.join(fold_save_path, 'best_model.pt'))
        
        metrics_save_path = os.path.join(fold_save_path, 'metrics.json')
        metrics_data = {
            'best_epoch': history['best_model_state']['epoch'],
            'best_metrics': history['best_model_state']['metrics'],
            'final_metrics': {
                'train': history['train_metrics'][-1],
                'val': history['val_metrics'][-1]
            },
            'learning_rate_history': history['lr'],
            'loss_history': {
                'train': history['train_loss'],
                'val': history['val_loss']
            }
        }
        
        with open(metrics_save_path, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        
        fold_histories.append(history)
        
        final_metrics = metrics_data['final_metrics']
        logger.info(f"\nFold {fold + 1} Results:")
        logger.info("Training Metrics:")
        for metric, value in final_metrics['train'].items():
            logger.info(f"  {metric}: {value:.6f}")
        logger.info("\nValidation Metrics:")
        for metric, value in final_metrics['val'].items():
            logger.info(f"  {metric}: {value:.6f}")
        best_val_loss = min(history['val_loss'])
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        logger.info(f"\nFold {fold + 1} Final Results:")
        logger.info(f"Final Train Loss: {final_train_loss:.6f}")
        logger.info(f"Final Val Loss: {final_val_loss:.6f}")
        logger.info(f"Best Val Loss: {best_val_loss:.6f}")
    
    avg_metrics = {
        'train': {},
        'val': {}
    }
    
    for metric in fold_histories[0]['train_metrics'][-1].keys():
        avg_metrics['train'][metric] = np.mean([h['train_metrics'][-1][metric] for h in fold_histories])
    for metric in fold_histories[0]['val_metrics'][-1].keys():
        avg_metrics['val'][metric] = np.mean([h['val_metrics'][-1][metric] for h in fold_histories])
    
    best_val_metrics = {
        metric: np.mean([h['best_model_state']['metrics'][metric] for h in fold_histories])
        for metric in fold_histories[0]['best_model_state']['metrics'].keys()
    }
    
    cv_results = {
        'avg_metrics': avg_metrics,
        'best_val_metrics': best_val_metrics,
        'fold_histories': [
            {
                'train_metrics': h['train_metrics'],
                'val_metrics': h['val_metrics'],
                'best_model_state': {
                    'epoch': h['best_model_state']['epoch'],
                    'metrics': h['best_model_state']['metrics']
                },
                'learning_rate_history': h['lr']
            }
            for h in fold_histories
        ]
    }
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f'cv_results_{timestamp}.json'
    
    with open(results_path, 'w') as f:
        json.dump(cv_results, f, indent=4)
    
    logger.info("\nCross-validation Results:")
    logger.info("\nTraining Metrics:")
    for metric, value in avg_metrics['train'].items():
        logger.info(f"Average {metric}: {value:.6f}")
    
    logger.info("\nValidation Metrics:")
    for metric, value in avg_metrics['val'].items():
        logger.info(f"Average {metric}: {value:.6f}")
    
    logger.info("\nBest Validation Metrics:")
    for metric, value in best_val_metrics.items():
        logger.info(f"Average Best {metric}: {value:.6f}")

    print("Configuration and data processor initialized.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred during training:")