"""
ISRO GNSS Monitoring Dashboard
-----------------------------
A comprehensive dashboard for monitoring and analyzing GNSS satellite data.
"""
import streamlit as st
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.dashboard.simple_auth import authenticator
from src.dashboard.pages import (
    dashboard_page, satellites_page, alerts_page, admin_panel
)

st.set_page_config(
    page_title="ISRO GNSS Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main content */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #1E3D8F;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #1E3D8F;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    
    .stButton>button:hover {
        background-color: #0d2a6b;
        color: white;
    }
    
    /* Metrics */
    .stMetric {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E3D8F;
        color: white;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    
    .streamlit-expanderContent {
        padding: 1rem;
        background-color: white;
        border-radius: 0 0 4px 4px;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    """Main application entry point."""
    if 'page' not in st.session_state:
        st.session_state.page = 'Dashboard'
    
    if not st.session_state.get('authenticated', False):
        if not authenticator.login():
            return
        
        st.rerun()
        return
    
    current_user = authenticator.get_current_user()
    if not current_user:
        st.error("Authentication failed. Please try again.")
        authenticator.logout()
        st.rerun()
        return
    
    user_role = current_user.get('role', 'viewer')
    
    with st.sidebar:
        st.title("Navigation")
        
        menu_options = ["Dashboard", "Satellites", "Alerts"]
        
        if user_role == 'admin':
            menu_options.append("Admin")
        
        selected_page = st.radio(
            "Go to",
            menu_options,
            index=menu_options.index(st.session_state.page) if st.session_state.page in menu_options else 0,
            key="page_selector"
        )
        
        if selected_page != st.session_state.page:
            st.session_state.page = selected_page
            st.rerun()
        
        st.markdown("---")
        
        st.markdown(f"""
            <div style='padding: 1rem; background-color: #f0f2f6; border-radius: 8px;'>
                <p style='margin: 0; font-weight: bold;'>{current_user.get('name', 'User')}</p>
                <p style='margin: 0; font-size: 0.8em; color: #666;'>{user_role.title()}</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Logout"):
            authenticator.logout()
        
        st.markdown("---")
        
        st.markdown("### System Status")
        st.markdown("""
            - **Last Update:** Just now
            - **Data Source:** ISRO GNSS Network
            - **Version:** 1.0.0
        """)
    
    if st.session_state.page == "Dashboard":
        dashboard_page.display()
    elif st.session_state.page == "Satellites":
        satellites_page.display()
    elif st.session_state.page == "Alerts":
        alerts_page.display()
    elif st.session_state.page == "Admin":
        if user_role == 'admin':
            admin_panel.display()
        else:
            st.error("You don't have permission to access the admin panel.")
            st.session_state.page = "Dashboard"
            st.rerun()
    else:
        st.warning("You don't have permission to access this page.")
        st.session_state.page = "Dashboard"
        st.rerun()

if __name__ == "__main__":
    main()
