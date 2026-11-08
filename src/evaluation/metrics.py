import numpy as np
import torch
from typing import Dict, Tuple, Optional
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor,
                     y_std: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_std: Standard deviation of predictions (for uncertainty quantification)
        
    Returns:
        Dictionary containing various metrics
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_std, torch.Tensor) and y_std is not None:
        y_std = y_std.detach().cpu().numpy()
    
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    smape = np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    
    non_zero_mask = np.abs(y_true) > 1e-8
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / 
                             np.abs(y_true[non_zero_mask]))) * 100
    else:
        mape = np.nan
    
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    if ss_tot < 1e-8:
        r2 = 1.0 if ss_res < 1e-8 else 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)
        r2 = max(r2, -1.0)
    
    nrmse = (rmse / (np.abs(y_mean) + 1e-8)) * 100
    
    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'nrmse': float(nrmse),
        'mape': float(mape) if not np.isnan(mape) else None,
        'smape': float(smape),
        'r2': float(r2)
    }
    
    if y_std is not None:
        crps = calculate_crps(y_true, y_pred, y_std)
        metrics['crps'] = float(crps)
        
        calibration = calculate_calibration(y_true, y_pred, y_std)
        metrics.update(calibration)
    
    return metrics

def calculate_crps(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> float:
    """Calculate Continuous Ranked Probability Score (CRPS).
    
    CRPS measures the quality of probabilistic forecasts, taking into account
    both accuracy and uncertainty quantification.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values (mean of the predictive distribution)
        y_std: Standard deviation of predictions
        
    Returns:
        CRPS score (lower is better)
    """
    z = (y_true - y_pred) / (y_std + 1e-8)
    
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    crps = y_std * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
    
    return float(np.mean(crps))

def calculate_calibration(y_true: np.ndarray, y_pred: np.ndarray,
                        y_std: np.ndarray) -> Dict[str, float]:
    """Calculate calibration metrics for uncertainty estimates.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_std: Standard deviation of predictions
        
    Returns:
        Dictionary containing calibration metrics
    """
    z = (y_true - y_pred) / (y_std + 1e-8)
    
    confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
    
    calibration_metrics = {}
    
    for level in confidence_levels:
        z_score = norm.ppf((1 + level) / 2)
        expected_coverage = level
        actual_coverage = np.mean(np.abs(z) <= z_score)
        
        calibration_error = np.abs(actual_coverage - expected_coverage)
        
        calibration_metrics[f'calibration_{int(level*100)}'] = float(actual_coverage)
        calibration_metrics[f'calibration_error_{int(level*100)}'] = float(calibration_error)
    
    calibration_metrics['mean_calibration_error'] = float(
        np.mean([calibration_metrics[f'calibration_error_{int(level*100)}']
                for level in confidence_levels])
    )
    
    return calibration_metrics

def evaluate_predictions(model: torch.nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       device: torch.device,
                       return_predictions: bool = False) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    """Evaluate model predictions with comprehensive metrics.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader containing validation/test data
        device: Device to run evaluation on
        return_predictions: Whether to return predictions array
        
    Returns:
        Tuple of (metrics_dict, predictions_dict)
    """
    model.eval()
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            
            all_y_true.append(y.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())
    
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    
    metrics = calculate_metrics(y_true, y_pred)
    
    if return_predictions:
        predictions = {
            'y_true': y_true,
            'y_pred': y_pred
        }
        return metrics, predictions
    
    return metrics, None