"""
Satellite Management Page

This module provides the UI and functionality for managing satellites in the ISRO GNSS Dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SatellitesPage:
    """Handles the display of satellite management features."""
    
    def __init__(self):
        """Initialize the SatellitesPage with sample data."""
        self.satellites = self._load_sample_data()
    
    def _load_sample_data(self):
        """Load sample satellite data."""
        return [
            {
                'id': 'IRNSS-1A',
                'name': 'IRNSS-1A',
                'type': 'Navigation',
                'status': 'Active',
                'launch_date': '2013-07-01',
                'orbit_type': 'Geosynchronous',
                'last_contact': (datetime.now() - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S'),
                'health': 'Good',
                'battery': '92%',
                'signal_strength': 'Strong'
            },
            {
                'id': 'IRNSS-1B',
                'name': 'IRNSS-1B',
                'type': 'Navigation',
                'status': 'Active',
                'launch_date': '2014-04-04',
                'orbit_type': 'Geosynchronous',
                'last_contact': (datetime.now() - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S'),
                'health': 'Good',
                'battery': '89%',
                'signal_strength': 'Strong'
            },
            {
                'id': 'IRNSS-1C',
                'name': 'IRNSS-1C',
                'type': 'Navigation',
                'status': 'Maintenance',
                'launch_date': '2014-10-16',
                'orbit_type': 'Geostationary',
                'last_contact': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
                'health': 'Warning',
                'battery': '75%',
                'signal_strength': 'Medium'
            }
        ]
    
    def display(self):
        """Display the satellite management interface."""
        st.title("Satellite Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_term = st.text_input("Search Satellites", "")
        
        with col2:
            status_filter = st.selectbox(
                "Filter by Status",
                ["All", "Active", "Inactive", "Maintenance"]
            )
        
        filtered_sats = self.satellites
        if search_term:
            filtered_sats = [sat for sat in filtered_sats 
                           if search_term.lower() in sat['name'].lower() 
                           or search_term.lower() in sat['id'].lower()]
        
        if status_filter != "All":
            filtered_sats = [sat for sat in filtered_sats 
                           if sat['status'] == status_filter]
        
        st.subheader("Satellite Fleet")
        
        for sat in filtered_sats:
            with st.expander(f"{sat['name']} - {sat['status']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**ID:** {sat['id']}")
                    st.markdown(f"**Type:** {sat['type']}")
                    st.markdown(f"**Orbit:** {sat['orbit_type']}")
                
                with col2:
                    st.markdown(f"**Health:** {sat['health']}")
                    st.markdown(f"**Battery:** {sat['battery']}")
                    st.markdown(f"**Last Contact:** {sat['last_contact']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"Refresh {sat['id']}", key=f"refresh_{sat['id']}"):
                        st.success(f"Refreshing data for {sat['name']}...")
                
                with col2:
                    if st.button(f"Diagnose {sat['id']}", key=f"diagnose_{sat['id']}"):
                        st.warning(f"Running diagnostics on {sat['name']}...")
                
                with col3:
                    if st.button(f"Command {sat['id']}", key=f"command_{sat['id']}"):
                        st.info(f"Opening command interface for {sat['name']}...")
        
        if st.session_state.get('user', {}).get('role') == 'admin':
            if st.button("Add New Satellite"):
                st.session_state['show_add_satellite'] = True
            
            if st.session_state.get('show_add_satellite', False):
                self._show_add_satellite_form()
    
    def _show_add_satellite_form(self):
        """Show the form to add a new satellite."""
        with st.form("add_satellite"):
            st.subheader("Add New Satellite")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sat_id = st.text_input("Satellite ID")
                name = st.text_input("Name")
                sat_type = st.selectbox("Type", ["Navigation", "Communication", "Earth Observation", "Weather"])
            
            with col2:
                status = st.selectbox("Status", ["Active", "Inactive", "Maintenance"])
                orbit_type = st.selectbox("Orbit Type", ["Geostationary", "Geosynchronous", "Low Earth Orbit", "Medium Earth Orbit"])
                launch_date = st.date_input("Launch Date")
            
            if st.form_submit_button("Add Satellite"):
                if sat_id and name:
                    new_sat = {
                        'id': sat_id,
                        'name': name,
                        'type': sat_type,
                        'status': status,
                        'launch_date': launch_date.strftime('%Y-%m-%d'),
                        'orbit_type': orbit_type,
                        'last_contact': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'health': 'Good',
                        'battery': '100%',
                        'signal_strength': 'Strong'
                    }
                    self.satellites.append(new_sat)
                    st.session_state['show_add_satellite'] = False
                    st.rerun()
                else:
                    st.error("Please fill in all required fields.")
            
            if st.form_submit_button("Cancel"):
                st.session_state['show_add_satellite'] = False
                st.rerun()

satellites_page = SatellitesPage()

def display():
    """Display the satellites page."""
    satellites_page.display()

if __name__ == "__main__":
    display()
