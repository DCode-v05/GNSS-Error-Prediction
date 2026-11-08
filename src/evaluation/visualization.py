"""
Visualization utilities for model evaluation.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import torch

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12

def plot_training_history(history: Dict[str, List[float]], 
                        save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot (if None, display the plot)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    if 'lr' in history:
        ax2.plot(history['lr'], label='Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.set_title('Learning Rate Schedule')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_predictions(y_true: np.ndarray, 
                    y_pred: np.ndarray, 
                    y_std: Optional[np.ndarray] = None,
                    timesteps: Optional[np.ndarray] = None,
                    title: str = "Predictions vs Actuals",
                    save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot predicted vs actual values with confidence intervals.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_std: Standard deviation of predictions (optional)
        timesteps: Time indices for x-axis (optional)
        title: Plot title
        save_path: Path to save the plot (if None, display the plot)
    """
    if timesteps is None:
        timesteps = np.arange(len(y_true))
    
    plt.figure(figsize=(14, 6))
    
    plt.plot(timesteps, y_true, 'b-', label='Actual', alpha=0.7, linewidth=2)
    
    plt.plot(timesteps, y_pred, 'r--', label='Predicted', alpha=0.8, linewidth=1.5)
    
    if y_std is not None:
        plt.fill_between(
            timesteps,
            y_pred - 1.96 * y_std,
            y_pred + 1.96 * y_std,
            color='r',
            alpha=0.2,
            label='95% Confidence Interval'
        )
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_error_distribution(errors: np.ndarray,
                          title: str = "Error Distribution",
                          save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot the distribution of prediction errors.
    
    Args:
        errors: Array of prediction errors (y_true - y_pred)
        title: Plot title
        save_path: Path to save the plot (if None, display the plot)
    """
    plt.figure(figsize=(10, 6))
    
    sns.histplot(errors, kde=True, stat="density", linewidth=0)
    
    x = np.linspace(errors.min(), errors.max(), 100)
    from scipy.stats import norm
    plt.plot(x, norm.pdf(x, errors.mean(), errors.std()), 'r-', lw=2, 
             label=f'Normal (μ={errors.mean():.4f}, σ={errors.std():.4f})')
    
    plt.xlabel('Prediction Error')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(feature_importance: Dict[str, float],
                          title: str = "Feature Importance",
                          top_n: Optional[int] = None,
                          save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot feature importance scores.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        title: Plot title
        top_n: Number of top features to show (if None, show all)
        save_path: Path to save the plot (if None, display the plot)
    """
    features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    if top_n is not None:
        features = features[:top_n]
    
    feature_names, importance = zip(*features)
    
    plt.figure(figsize=(10, max(6, len(feature_names) * 0.4)))
    y_pos = np.arange(len(feature_names))
    
    plt.barh(y_pos, importance, align='center', alpha=0.7)
    plt.yticks(y_pos, feature_names)
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_correlation_matrix(data: pd.DataFrame,
                          title: str = "Feature Correlation Matrix",
                          figsize: Tuple[int, int] = (12, 10),
                          save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot a correlation matrix heatmap.
    
    Args:
        data: DataFrame containing the data
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the plot (if None, display the plot)
    """
    corr = data.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    plt.figure(figsize=figsize)
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=True,
        fmt=".2f"
    )
    
    plt.title(title, pad=20)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
