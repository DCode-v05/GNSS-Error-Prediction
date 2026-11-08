"""Data visualization components for the ISRO GNSS Dashboard."""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataVisualization:
    def __init__(self):
        self.color_scale = px.colors.qualitative.Plotly
        
    def plot_time_series(self, data, x_col, y_cols, title="", y_title="", x_title="Time"):
        """Generic time series plot.
        
        Args:
            data (pd.DataFrame): DataFrame containing the data
            x_col (str): Column name for x-axis (time)
            y_cols (list): List of column names for y-axes
            title (str): Plot title
            y_title (str): Y-axis title
            x_title (str): X-axis title
        """
        if data.empty or not y_cols:
            st.warning("No data available for time series plot")
            return
            
        fig = go.Figure()
        
        for i, col in enumerate(y_cols):
            if col not in data.columns:
                continue
                
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[col],
                mode='lines',
                name=col,
                line=dict(color=self.color_scale[i % len(self.color_scale)]),
                hovertemplate='%{y:.4f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_error_metrics(self, metrics_data):
        """Plot error metrics for satellite predictions.
        
        Args:
            metrics_data (dict): Dictionary containing error metrics
        """
        if not metrics_data:
            st.warning("No error metrics available")
            return
            
        df = pd.DataFrame(metrics_data)
        
        fig = go.Figure()
        
        metrics = ['mae', 'rmse', 'max_error']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for metric, color in zip(metrics, colors):
            if metric in df.columns:
                fig.add_trace(go.Bar(
                    x=df.index,
                    y=df[metric],
                    name=metric.upper(),
                    marker_color=color,
                    text=df[metric].round(4),
                    textposition='auto'
                ))
        
        fig.update_layout(
            title='Prediction Error Metrics',
            barmode='group',
            xaxis_title='Satellite',
            yaxis_title='Error Value',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_heatmap(self, data, x, y, z, title=""):
        """Plot a heatmap.
        
        Args:
            data (pd.DataFrame): Input data
            x (str): Column for x-axis
            y (str): Column for y-axis
            z (str): Column for color scale
            title (str): Plot title
        """
        if data.empty or x not in data.columns or y not in data.columns or z not in data.columns:
            st.warning("Insufficient data for heatmap")
            return
            
        fig = px.density_heatmap(
            data, 
            x=x, 
            y=y, 
            z=z,
            title=title,
            color_continuous_scale='Viridis',
            labels={x: x.title(), y: y.title(), z: z.title()}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_histogram(self, data, column, title="", nbins=20):
        """Plot a histogram.
        
        Args:
            data (pd.Series or list): Input data
            title (str): Plot title
            nbins (int): Number of bins
        """
        if data is None or len(data) == 0:
            st.warning("No data available for histogram")
            return
            
        fig = px.histogram(
            x=data,
            nbins=nbins,
            title=title,
            labels={'x': column, 'y': 'Count'},
            color_discrete_sequence=[self.color_scale[0]]
        )
        
        fig.update_layout(
            bargap=0.1,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_metrics_grid(self, metrics, columns=3):
        """Display a grid of metric cards.
        
        Args:
            metrics (list): List of (label, value, delta, delta_type) tuples
            columns (int): Number of columns in the grid
        """
        if not metrics:
            return
            
        cols = st.columns(columns)
        
        for i, (label, value, delta, delta_type) in enumerate(metrics):
            with cols[i % columns]:
                st.metric(
                    label=label,
                    value=value,
                    delta=delta,
                    delta_color=delta_type if delta is not None else None
                )

data_viz = DataVisualization()
