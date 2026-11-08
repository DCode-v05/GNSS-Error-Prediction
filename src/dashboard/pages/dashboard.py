"""Main dashboard page for the ISRO GNSS Monitoring System."""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.dashboard.components.header import Header
from src.dashboard.components.satellite_visualization import SatelliteVisualization
from src.dashboard.components.data_visualization import data_viz

class DashboardPage:
    def __init__(self):
        self.header = Header()
        self.sat_viz = SatelliteVisualization()
        
        self._init_session_state()
        
    def _init_session_state(self):
        """Initialize required session state variables."""
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
            
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
            
        if 'satellites' not in st.session_state:
            st.session_state.satellites = []
            
        if 'user_prefs' not in st.session_state:
            st.session_state.user_prefs = {
                'auto_refresh': True,
                'refresh_interval': 300,
                'theme': 'light',
                'notifications': True
            }
        
    def _generate_sample_data(self):
        """Generate comprehensive satellite data for the dashboard."""
        now = datetime.now()
        
        satellites = [
            {
                'id': 'IRNSS-1A',
                'x': np.cos(0) * 1.8,
                'y': np.sin(0) * 1.8,
                'z': 0.5,
                'altitude': 35786,
                'inclination': 29,
                'phase': 0,
                'status': 'Operational',
                'status_color': 'green',
                'last_contact': (now - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S'),
                'battery': 94,
                'signal_strength': 'Strong',
                'uplink': 'Normal',
                'downlink': 'Normal',
                'temperature': 32,
                'last_maintenance': '2023-11-15',
                'next_maintenance': '2024-05-15'
            },
            {
                'id': 'IRNSS-1B',
                'x': np.cos(np.pi/3) * 1.8,
                'y': np.sin(np.pi/3) * 1.8,
                'z': 0.5,
                'altitude': 35900,
                'inclination': 29,
                'phase': np.pi/3,
                'status': 'Operational',
                'status_color': 'green',
                'last_contact': (now - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S'),
                'battery': 92,
                'signal_strength': 'Strong',
                'uplink': 'Normal',
                'downlink': 'Normal',
                'temperature': 31,
                'last_maintenance': '2023-10-20',
                'next_maintenance': '2024-04-20'
            },
            {
                'id': 'IRNSS-1C',
                'x': np.cos(2*np.pi/3) * 1.8,
                'y': np.sin(2*np.pi/3) * 1.8,
                'z': 0.5,
                'altitude': 35800,
                'inclination': 5,
                'phase': 2*np.pi/3,
                'status': 'Operational',
                'status_color': 'green',
                'last_contact': (now - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S'),
                'battery': 96,
                'signal_strength': 'Strong',
                'uplink': 'Normal',
                'downlink': 'Normal',
                'temperature': 30,
                'last_maintenance': '2024-01-10',
                'next_maintenance': '2024-07-10'
            },
            {
                'id': 'IRNSS-1D',
                'x': np.cos(np.pi) * 1.8,
                'y': np.sin(np.pi) * 1.8,
                'z': 0.5,
                'altitude': 35850,
                'inclination': 31,
                'phase': np.pi,
                'status': 'Operational',
                'status_color': 'green',
                'last_contact': now.strftime('%Y-%m-%d %H:%M:%S'),
                'battery': 93,
                'signal_strength': 'Strong',
                'uplink': 'Normal',
                'downlink': 'Normal',
                'temperature': 33,
                'last_maintenance': '2023-12-05',
                'next_maintenance': '2024-06-05'
            },
            {
                'id': 'IRNSS-1E',
                'x': np.cos(4*np.pi/3) * 1.8,
                'y': np.sin(4*np.pi/3) * 1.8,
                'z': 0.5,
                'altitude': 35700,
                'inclination': 29,
                'phase': 4*np.pi/3,
                'status': 'Maintenance',
                'status_color': 'orange',
                'last_contact': (now - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
                'battery': 85,
                'signal_strength': 'Weak',
                'uplink': 'Degraded',
                'downlink': 'Degraded',
                'temperature': 38,
                'last_maintenance': '2023-09-20',
                'next_maintenance': '2024-02-15'
            },
            {
                'id': 'IRNSS-1F',
                'x': np.cos(5*np.pi/3) * 1.8,
                'y': np.sin(5*np.pi/3) * 1.8,
                'z': 0.5,
                'altitude': 35800,
                'inclination': 5,
                'phase': 5*np.pi/3,
                'status': 'Operational',
                'status_color': 'green',
                'last_contact': (now - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S'),
                'battery': 91,
                'signal_strength': 'Strong',
                'uplink': 'Normal',
                'downlink': 'Normal',
                'temperature': 31,
                'last_maintenance': '2024-01-25',
                'next_maintenance': '2024-07-25'
            },
            {
                'id': 'IRNSS-1G',
                'x': np.cos(2*np.pi) * 1.8,
                'y': np.sin(2*np.pi) * 1.8,
                'z': 0.5,
                'altitude': 35900,
                'inclination': 30,
                'phase': 2*np.pi,
                'status': 'Operational',
                'status_color': 'green',
                'last_contact': (now - timedelta(minutes=45)).strftime('%Y-%m-%d %H:%M:%S'),
                'battery': 95,
                'signal_strength': 'Strong',
                'uplink': 'Normal',
                'downlink': 'Normal',
                'temperature': 30,
                'last_maintenance': '2023-11-30',
                'next_maintenance': '2024-05-30'
            }
        ]
        
        timestamps = [now - timedelta(minutes=30-i) for i in range(30)]
        time_series_data = pd.DataFrame({
            'timestamp': timestamps,
            'IRNSS-1A': np.sin(np.linspace(0, 10, 30)) * 0.5 + np.random.normal(0, 0.05, 30),
            'IRNSS-1B': np.cos(np.linspace(0, 10, 30)) * 0.5 + np.random.normal(0, 0.05, 30),
            'IRNSS-1C': np.sin(np.linspace(0, 10, 30) + 1) * 0.5 + np.random.normal(0, 0.05, 30),
            'IRNSS-1D': np.cos(np.linspace(0, 10, 30) + 1) * 0.5 + np.random.normal(0, 0.05, 30),
            'IRNSS-1E': np.sin(np.linspace(0, 10, 30) + 2) * 0.5 + np.random.normal(0, 0.1, 30),
            'IRNSS-1F': np.cos(np.linspace(0, 10, 30) + 2) * 0.5 + np.random.normal(0, 0.05, 30),
            'IRNSS-1G': np.sin(np.linspace(0, 10, 30) + 3) * 0.5 + np.random.normal(0, 0.05, 30)
        })
        
        error_metrics = {
            'IRNSS-1A': {'mae': 0.12, 'rmse': 0.15, 'max_error': 0.25},
            'IRNSS-1B': {'mae': 0.18, 'rmse': 0.22, 'max_error': 0.35},
            'IRNSS-1C': {'mae': 0.10, 'rmse': 0.13, 'max_error': 0.20},
            'IRNSS-1D': {'mae': 0.15, 'rmse': 0.18, 'max_error': 0.30},
            'IRNSS-1E': {'mae': 0.25, 'rmse': 0.30, 'max_error': 0.45},
            'IRNSS-1F': {'mae': 0.14, 'rmse': 0.17, 'max_error': 0.28},
            'IRNSS-1G': {'mae': 0.11, 'rmse': 0.14, 'max_error': 0.22}
        }
        
        active_alerts = sum(1 for sat in satellites if sat['status'] != 'Operational')
        
        return {
            'satellites': satellites,
            'time_series': time_series_data,
            'error_metrics': error_metrics,
            'status': {
                'satellites_tracked': len(satellites),
                'active_alerts': active_alerts,
                'system_status': 'Nominal' if active_alerts == 0 else 'Degraded',
                'data_freshness_sec': 5
            }
        }
    
    def display(self):
        """Display the main dashboard."""
        self.header.display()
        
        data = self._generate_sample_data()
        
        self.header.display_status_bar(data.get('status', {}))
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.container():
                st.subheader("Satellite Orbits")
                self.sat_viz.plot_satellite_positions(data.get('satellites', []))
            
            with st.container():
                st.subheader("Position Errors Over Time")
                if 'time_series' in data and not data['time_series'].empty:
                    data_viz.plot_time_series(
                        data['time_series'],
                        x_col='timestamp',
                        y_cols=[col for col in data['time_series'].columns if col != 'timestamp'],
                        y_title='Error (m)'
                    )
                else:
                    st.warning("No time series data available")
        
        with col2:
            with st.container():
                st.subheader("System Metrics")
                metrics = [
                    ("Uptime", "12d 06:45:32", "+2h 15m", "normal"),
                    ("Data Rate", "1.2 MB/s", "-0.1 MB/s", "inverse"),
                    ("Active Users", "8", "+2", "off"),
                    ("Avg. Error", "0.15m", "-0.02m", "normal")
                ]
                data_viz.display_metrics_grid(metrics, columns=2)
            
            with st.container():
                st.subheader("Prediction Errors")
                if 'error_metrics' in data and data['error_metrics']:
                    data_viz.plot_error_metrics(data['error_metrics'])
                else:
                    st.warning("No error metrics available")
            
            with st.container():
                st.subheader("Recent Alerts")
                
                now = datetime.now()
                
                alert_types = {
                    'warning': {'icon': '⚠️', 'color': 'orange'},
                    'error': {'icon': '❌', 'color': 'red'},
                    'success': {'icon': '✅', 'color': 'green'},
                    'info': {'icon': 'ℹ️', 'color': 'blue'},
                    'critical': {'icon': '🚨', 'color': 'darkred'}
                }
                
                alerts = [
                    {
                        "time": now - timedelta(minutes=2),
                        "satellite": "IRNSS-1E",
                        "message": "High temperature threshold exceeded (38°C)",
                        "level": "warning",
                        "details": "Satellite temperature has exceeded operational limits. Monitoring for thermal anomalies.",
                        "action_required": "Monitor temperature trends"
                    },
                    {
                        "time": now - timedelta(minutes=15),
                        "satellite": "IRNSS-1B",
                        "message": "Signal strength fluctuation detected",
                        "level": "warning",
                        "details": "Signal strength variation of ±2dB observed. Still within operational limits.",
                        "action_required": "Monitor signal stability"
                    },
                    {
                        "time": now - timedelta(minutes=30),
                        "satellite": "IRNSS-1A",
                        "message": "Minor orbit adjustment completed",
                        "level": "info",
                        "details": "Scheduled station-keeping maneuver completed successfully.",
                        "action_required": "None"
                    },
                    {
                        "time": now - timedelta(hours=1),
                        "satellite": "IRNSS-1D",
                        "message": "New ephemeris data received",
                        "level": "info",
                        "details": "Updated orbital parameters received and processed.",
                        "action_required": "None"
                    },
                    {
                        "time": now - timedelta(hours=3),
                        "message": "Daily system health check completed",
                        "level": "success",
                        "details": "All systems operating within normal parameters. No issues detected.",
                        "action_required": "None"
                    },
                    {
                        "time": now - timedelta(hours=6),
                        "satellite": "IRNSS-1F",
                        "message": "Battery charge cycle completed",
                        "level": "info",
                        "details": "Battery maintenance cycle completed. Battery health at 98%.",
                        "action_required": "None"
                    },
                    {
                        "time": now - timedelta(days=1),
                        "satellite": "IRNSS-1C",
                        "message": "Antenna calibration successful",
                        "level": "success",
                        "details": "Routine antenna calibration completed within expected parameters.",
                        "action_required": "None"
                    },
                    {
                        "time": now - timedelta(days=2),
                        "satellite": "Ground Station",
                        "message": "Backup power system test",
                        "level": "info",
                        "details": "Scheduled backup power system test completed successfully.",
                        "action_required": "None"
                    }
                ]
                
                alerts.sort(key=lambda x: x['time'], reverse=True)
                
                for alert in alerts[:10]:
                    alert_type = alert.get('level', 'info')
                    alert_config = alert_types.get(alert_type, alert_types['info'])
                    
                    time_diff = now - alert['time']
                    if time_diff < timedelta(minutes=1):
                        time_str = "Just now"
                    elif time_diff < timedelta(hours=1):
                        minutes = int(time_diff.total_seconds() / 60)
                        time_str = f"{minutes} minute{'s' if minutes > 1 else ''} ago"
                    elif time_diff < timedelta(days=1):
                        hours = int(time_diff.total_seconds() / 3600)
                        time_str = f"{hours} hour{'s' if hours > 1 else ''} ago"
                    else:
                        days = time_diff.days
                        time_str = f"{days} day{'s' if days > 1 else ''} ago"
                    
                    with st.expander(
                        f"{alert_config['icon']} {alert.get('message', 'Alert')} - {time_str}",
                        expanded=alert_type in ['error', 'critical']
                    ):
                        cols = st.columns([1, 3])
                        with cols[0]:
                            st.markdown(f"**Time:** {alert['time'].strftime('%Y-%m-%d %H:%M:%S')}")
                            if 'satellite' in alert:
                                st.markdown(f"**Satellite:** {alert['satellite']}")
                            st.markdown(f"**Level:** :{alert_config['color']}[{alert_type.upper()}]")
                        
                        with cols[1]:
                            st.markdown(f"**Details:** {alert.get('details', 'No additional details available.')}")
                            if alert.get('action_required') and alert['action_required'].lower() != 'none':
                                st.markdown(f"**Action Required:** :red[{alert['action_required']}]")
                        
                        if alert_type not in ['error', 'critical']:
                            if st.button("Dismiss", key=f"dismiss_{alert['time'].timestamp()}"):
                                alerts.remove(alert)
                                st.rerun()
        
        if st.button("🔄 Refresh Data"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
            
        current_time = datetime.now()
        last_refresh = st.session_state.get('last_refresh', current_time)
        
        if (current_time - last_refresh).total_seconds() > 300:
            st.session_state.last_refresh = current_time
            st.rerun()

dashboard_page = DashboardPage()

__all__ = ['dashboard_page', 'DashboardPage']
