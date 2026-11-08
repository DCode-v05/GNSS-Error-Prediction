import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any

sns.set_theme(style="whitegrid")
import json
import requests
from pathlib import Path
from typing import Dict, List, Tuple

def load_model_metrics(history_path: str) -> dict:
    """
    Load model training history from JSON file.
    
    Args:
        history_path: Path to the history JSON file
        
    Returns:
        Dictionary containing training metrics
    """
    with open(history_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def generate_sample_data(n_samples: int = 100) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate sample GNSS data for the dashboard.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (DataFrame, feature_columns)
    """
    np.random.seed(42)
    
    end_time = pd.Timestamp.now()
    start_time = end_time - pd.Timedelta(days=1)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=n_samples)
    
    data = {
        'utc_time': timestamps,
        'sat_id': np.random.choice(['G01', 'G02', 'G03'], size=n_samples),
        'clock_error': np.random.normal(0, 1, n_samples).cumsum(),
        'ephemeris_error': np.random.normal(0, 0.5, n_samples).cumsum(),
        'elevation': np.random.uniform(5, 90, n_samples),
        'azimuth': np.random.uniform(0, 360, n_samples),
        'snr': np.random.normal(45, 5, n_samples).clip(30, 60)
    }
    
    df = pd.DataFrame(data)
    
    feature_columns = [col for col in df.select_dtypes(include=['float64', 'int64']).columns 
                      if col != 'utc_time']
    
    return df, feature_columns
    
    return df, feature_columns

def normalize_sequence(sequence: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    """
    Normalize sequence using min-max scaling.
    
    Args:
        sequence: Input sequence
        min_vals: Minimum values for each feature
        max_vals: Maximum values for each feature
        
    Returns:
        Normalized sequence
    """
    return (sequence - min_vals) / (max_vals - min_vals + 1e-8)

def plot_predictions(predictions: Dict[str, float]):
    """
    Plot predictions for each error type.
    
    Args:
        predictions: Dictionary mapping error types to predicted values
    """
    plt.figure(figsize=(10, 6))
    plt.bar(predictions.keys(), predictions.values())
    plt.xticks(rotation=45)
    plt.xlabel('Error Type')
    plt.ylabel('Error (meters)')
    plt.title('GNSS Error Predictions')
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

def plot_error_distribution(df: pd.DataFrame, error_type: str):
    """
    Plot error distribution histogram.
    
    Args:
        df: DataFrame containing error values
        error_type: Type of error to plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df[error_type], bins=30, density=True, alpha=0.7)
    plt.xlabel(f'{error_type} (meters)')
    plt.ylabel('Density')
    plt.title(f'{error_type} Distribution')
    st.pyplot(plt)
    plt.close()

def make_prediction_request(sequence: List[List[float]], feature_columns: List[str]) -> Dict[str, float]:
    """
    Make prediction request to the API.
    
    Args:
        sequence: Input sequence for prediction
        feature_columns: Names of input features
        
    Returns:
        Dictionary containing predictions for each error type
    """
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={
                "sequence": sequence,
                "feature_columns": feature_columns
            }
        )
        response.raise_for_status()
        return response.json()['predictions']
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        raise

def add_custom_css():
    st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stSelectbox, .stSlider, .stDateInput {
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .stPlotlyChart {
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="GNSS Error Prediction Dashboard",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    add_custom_css()
    
    st.title("🛰️ GNSS Error Prediction Dashboard")
    st.markdown("---")
    
    try:
        df, feature_columns = generate_sample_data(n_samples=100)
        st.sidebar.success("Using generated sample data")
        
        st.sidebar.subheader("Sample Data Preview")
        st.sidebar.dataframe(df.head())
        
        st.sidebar.header("Filters")
        
        all_satellites = df['sat_id'].unique().tolist()
        selected_satellites = st.sidebar.multiselect(
            'Select Satellites',
            options=all_satellites,
            default=all_satellites
        )
        
        min_time = df['utc_time'].min().to_pydatetime()
        max_time = df['utc_time'].max().to_pydatetime()
        
        time_range = st.sidebar.slider(
            'Select Time Range',
            min_value=min_time,
            max_value=max_time,
            value=(min_time, max_time)
        )
        
        filtered_df = df[
            (df['sat_id'].isin(selected_satellites)) &
            (df['utc_time'] >= time_range[0]) &
            (df['utc_time'] <= time_range[1])
        ]
        
        st.subheader("📊 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='metric-card'><h3>Total Samples</h3><h2>{:,}</h2></div>".format(len(filtered_df)), unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='metric-card'><h3>Satellites</h3><h2>{}</h2></div>".format(filtered_df['sat_id'].nunique()), unsafe_allow_html=True)
        with col3:
            avg_clock_error = filtered_df['clock_error'].mean()
            st.markdown(f"<div class='metric-card'><h3>Avg Clock Error</h3><h2>{avg_clock_error:.4f}</h2></div>", unsafe_allow_html=True)
        with col4:
            avg_eph_error = filtered_df['ephemeris_error'].mean()
            st.markdown(f"<div class='metric-card'><h3>Avg Ephemeris Error</h3><h2>{avg_eph_error:.4f}</h2></div>", unsafe_allow_html=True)
            
        st.caption(f"Showing data from {time_range[0].strftime('%Y-%m-%d %H:%M')} to {time_range[1].strftime('%Y-%m-%d %H:%M')}")
            
        chart_df = filtered_df.copy()
        chart_df['time_str'] = chart_df['utc_time'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.subheader("📈 Error Trends Over Time")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Error Trends", 
            "📡 Satellite Comparison", 
            "📉 Error Distribution", 
            "🔍 Correlation Analysis"
        ])
        
        with tab1:
            st.markdown("### Clock and Ephemeris Errors")
            chart_data = chart_df[['time_str', 'clock_error', 'ephemeris_error']].set_index('time_str')
            st.line_chart(chart_data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Max Clock Error", f"{chart_df['clock_error'].max():.4f}")
                st.metric("Min Clock Error", f"{chart_df['clock_error'].min():.4f}")
            with col2:
                st.metric("Max Ephemeris Error", f"{chart_df['ephemeris_error'].max():.4f}")
                st.metric("Min Ephemeris Error", f"{chart_df['ephemeris_error'].min():.4f}")
        
        with tab2:
            st.markdown("### Error by Satellite")
            sat_errors = chart_df.groupby('sat_id')[['clock_error', 'ephemeris_error']].mean().reset_index()
            
            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(sat_errors, x='sat_id', y='clock_error')
                st.caption("Average Clock Error by Satellite")
            with col2:
                st.bar_chart(sat_errors, x='sat_id', y='ephemeris_error')
                st.caption("Average Ephemeris Error by Satellite")
        
        with tab3:
            st.markdown("### Error Distribution")
            
            col1, col2 = st.columns(2)
            with col1:
                st.area_chart(chart_df['clock_error'])
                st.caption("Clock Error Distribution")
            with col2:
                st.area_chart(chart_df['ephemeris_error'])
                st.caption("Ephemeris Error Distribution")
            
            st.markdown("#### Error Histograms")
            hist_bins = st.slider("Number of bins", 5, 50, 20)
            
            fig1, ax1 = plt.subplots()
            sns.histplot(data=chart_df, x='clock_error', bins=hist_bins, kde=True, ax=ax1)
            ax1.set_title('Clock Error Distribution')
            st.pyplot(fig1)
            
            fig2, ax2 = plt.subplots()
            sns.histplot(data=chart_df, x='ephemeris_error', bins=hist_bins, kde=True, ax=ax2)
            ax2.set_title('Ephemeris Error Distribution')
            st.pyplot(fig2)
        
        with tab4:
            st.markdown("### Feature Correlations")
            
            numeric_cols = chart_df.select_dtypes(include=['float64', 'int64']).columns
            corr = chart_df[numeric_cols].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            st.pyplot(plt)
            
            st.markdown("#### Strongest Correlations")
            corr_pairs = corr.unstack().sort_values(ascending=False)
            corr_pairs = corr_pairs[corr_pairs < 1]
            st.dataframe(corr_pairs.head(10).reset_index().rename(columns={'level_0': 'Feature 1', 'level_1': 'Feature 2', 0: 'Correlation'}))
        
    except Exception as e:
        st.error(f"Error in dashboard: {str(e)}")
        st.warning("Falling back to sample data generation...")
        
        try:
            df, feature_columns = generate_sample_data(n_samples=50)
            
            st.subheader("Sample Data Loaded")
            st.write("Using generated sample data as fallback")
            
            preview_df = df.copy()
            preview_df['utc_time'] = preview_df['utc_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(preview_df.head())
            
            chart_df = df.copy()
            
            chart_df['time_str'] = chart_df['utc_time'].dt.strftime('%H:%M')
            
            st.subheader("Clock Error Over Time")
            st.line_chart(data=chart_df.set_index('time_str')['clock_error'])
            
            st.subheader("Ephemeris Error Over Time")
            st.line_chart(data=chart_df.set_index('time_str')['ephemeris_error'])
            
        except Exception as fallback_error:
            st.error(f"Failed to generate sample data: {str(fallback_error)}")
        
        st.info("The dashboard is displaying sample GNSS error data. To customize this dashboard with your own data, please update the 'generate_sample_data' function in app.py.")
    
    st.write("\n" * 5)
    
    st.markdown("---")
    st.markdown("GNSS Error Prediction Dashboard • [GitHub](https://github.com/yourusername/gnss-error-prediction)")
if __name__ == "__main__":
    main()