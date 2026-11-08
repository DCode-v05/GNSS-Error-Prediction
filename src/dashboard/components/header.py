"""Header component for the ISRO GNSS Dashboard."""
import streamlit as st
from datetime import datetime
import pytz

class Header:
    def __init__(self):
        self.timezone = pytz.timezone('Asia/Kolkata')
        self.refresh_rate = 60
        
    def display(self):
        """Display the dashboard header."""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.image("https://www.isro.gov.in/sites/default/files/footer-logo.png", 
                    width=100)
            
        with col2:
            st.title("ISRO GNSS Monitoring Dashboard")
            current_time = datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M:%S %Z")
            st.caption(f"Last Updated: {current_time}")
            
        with col3:
            if 'name' in st.session_state:
                st.markdown(f"""
                    <div style='text-align: right;'>
                        <span style='font-size: 0.9em;'>Welcome,</span><br>
                        <strong>{st.session_state['name']}</strong><br>
                        <span style='font-size: 0.8em; color: #666;'>{st.session_state.get('role', 'viewer').title()}</span>
                    </div>
                """, unsafe_allow_html=True)
                
        st.markdown("---")
        
    def display_status_bar(self, status_data):
        """Display status bar with key metrics.
        
        Args:
            status_data (dict): Dictionary containing status information
        """
        if not status_data:
            return
            
        cols = st.columns(4)
        metrics = [
            ("Satellites Tracked", status_data.get('satellites_tracked', 0), "🛰️"),
            ("Active Alerts", status_data.get('active_alerts', 0), "⚠️"),
            ("System Status", status_data.get('system_status', 'Unknown'), "✅"),
            ("Data Freshness", f"{status_data.get('data_freshness_sec', 0)}s", "⏱️")
        ]
        
        for idx, (label, value, icon) in enumerate(metrics):
            with cols[idx]:
                st.metric(label, f"{icon} {value}")
                
        st.markdown("---")
