import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import logging
from src.data.data_processor import GNSSDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_target_distribution(data: np.ndarray, save_dir: str = 'results/analysis'):
    """Analyze target variable distribution and suggest transformations."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    stats_dict = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'skew': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
        'min': np.min(data),
        'max': np.max(data)
    }
    
    logger.info("\nTarget Variable Statistics:")
    for stat, value in stats_dict.items():
        logger.info(f"{stat}: {value:.6f}")
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    sns.histplot(data, bins=50, kde=True)
    plt.title('Original Distribution')
    plt.xlabel('Value')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 2)
    log_data = np.log1p(np.abs(data)) * np.sign(data)
    sns.histplot(log_data, bins=50, kde=True)
    plt.title('Log Transformed Distribution')
    plt.xlabel('Log(|x| + 1) * sign(x)')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 3)
    plt.boxplot(data)
    plt.title('Box Plot')
    plt.ylabel('Value')
    
    plt.subplot(2, 2, 4)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'target_distribution.png')
    plt.close()
    
    normality_tests = {
        'Shapiro-Wilk': stats.shapiro(data[:5000]),
        'D\'Agostino-Pearson': stats.normaltest(data),
        'Kolmogorov-Smirnov': stats.kstest(data, 'norm')
    }
    
    logger.info("\nNormality Tests:")
    for test, result in normality_tests.items():
        logger.info(f"{test} - statistic: {result[0]:.6f}, p-value: {result[1]:.6f}")
    
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = np.sum((data < lower_bound) | (data > upper_bound))
    outlier_percentage = (outliers / len(data)) * 100
    
    logger.info(f"\nOutlier Analysis:")
    logger.info(f"Number of outliers: {outliers}")
    logger.info(f"Percentage of outliers: {outlier_percentage:.2f}%")
    logger.info(f"IQR range: [{lower_bound:.6f}, {upper_bound:.6f}]")
    
    logger.info("\nTransformation Suggestions:")
    if abs(stats_dict['skew']) > 1:
        logger.info("- High skewness detected, consider log transformation")
    if stats_dict['kurtosis'] > 3:
        logger.info("- High kurtosis detected, consider power transformation")
    if outlier_percentage > 5:
        logger.info("- High percentage of outliers, consider robust scaling")
    
    return stats_dict, normality_tests

def main():
    data_processor = GNSSDataProcessor()
    df = data_processor.load_data('data/raw/DATA_GEO_Train.csv')
    
    error_cols = [col for col in df.columns if 'error' in col.lower()]
    
    for col in error_cols:
        logger.info(f"\nAnalyzing column: {col}")
        data = df[col].values
        stats_dict, _ = analyze_target_distribution(data, save_dir=f'results/analysis/{col}')
        
        stats_df = pd.DataFrame([stats_dict])
        stats_df.to_csv(f'results/analysis/{col}/statistics.csv', index=False)

if __name__ == '__main__':
    main()