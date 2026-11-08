"""
Authentication module for the ISRO GNSS Dashboard.

This module provides user authentication functionality using Streamlit-Authenticator.
It handles user login, session management, and role-based access control.
"""
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import os
import hashlib
import binascii

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
CONFIG_FILE = CONFIG_DIR / "users.yaml"

def hash_password(password):
    """Hash a password for storing."""
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
    pwdhash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), salt, 100000)
    pwdhash = binascii.hexlify(pwdhash)
    return (salt + pwdhash).decode('ascii')

class Authenticator:
    """Handles user authentication and session management."""
    
    def __init__(self):
        """Initialize the authenticator and load user configuration."""
        self.authenticator = None
        self.config = {}
        self._load_config()
        self._initialize_authenticator()
    
    def _load_config(self):
        """Load user configuration from YAML file."""
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            
            if not CONFIG_FILE.exists():
                self._create_default_config()
                return
            
            with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
                self.config = yaml.load(file, Loader=SafeLoader) or {}
                
            if not self.config or 'credentials' not in self.config:
                raise ValueError("Invalid configuration: Missing 'credentials' section")
                
        except Exception as e:
            st.error(f"Error loading configuration: {str(e)}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create a default configuration with default users."""
        print("Creating default configuration...")
        
        self.config = {
            'credentials': {
                'usernames': {
                    'admin': {
                        'email': 'admin@isro.gov.in',
                        'name': 'Admin User',
                        'password': hash_password('admin123'),
                        'role': 'admin'
                    },
                    'operator': {
                        'email': 'operator@isro.gov.in',
                        'name': 'Operator User',
                        'password': hash_password('operator123'),
                        'role': 'operator'
                    },
                    'viewer': {
                        'email': 'viewer@isro.gov.in',
                        'name': 'Viewer User',
                        'password': hash_password('viewer123'),
                        'role': 'viewer'
                    }
                }
            },
            'cookie': {
                'name': 'auth_cookie',
                'key': 'default_signature_key',
                'expiry_days': 1
            },
            'preauthorized': None
        }
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as file:
            yaml.dump(self.config, file, default_flow_style=False)
    
    def _initialize_authenticator(self):
        """Initialize the Streamlit authenticator."""
        try:
            credentials = {
                'usernames': {}
            }
            
            for username, user_data in self.config['credentials']['usernames'].items():
                credentials['usernames'][username] = {
                    'email': user_data['email'],
                    'name': user_data['name'],
                    'password': user_data['password']
                }
            
            import time
            unique_key = f"auth_{int(time.time())}"
            
            self.authenticator = stauth.Authenticate(
                credentials=credentials,
                cookie_name=self.config['cookie']['name'],
                key=unique_key,
                cookie_expiry_days=self.config['cookie']['expiry_days']
            )
            
        except Exception as e:
            st.error(f"Failed to initialize authenticator: {str(e)}")
            raise
    
    def login(self):
        """Handle user login."""
        if not self.authenticator:
            st.error("Authentication system not properly initialized.")
            return False
        
        try:
            name, authentication_status, username = self.authenticator.login('Login', 'main')
            
            if authentication_status:
                st.session_state['authentication_status'] = True
                st.session_state['username'] = username
                st.session_state['name'] = name
                st.rerun()
            elif authentication_status is False:
                st.error('Username/password is incorrect')
            elif authentication_status is None:
                st.warning('Please enter your username and password')
                
            return authentication_status
            
        except Exception as e:
            st.error(f"An error occurred during login: {str(e)}")
            return False
    
    def logout(self):
        """Handle user logout."""
        if st.sidebar.button("Logout"):
            for key in ['authentication_status', 'username', 'name']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    def get_current_user(self):
        """Get the current authenticated user's information."""
        if not st.session_state.get('authentication_status'):
            return None
            
        username = st.session_state.get('username')
        if not username:
            return None
            
        user_data = self.config['credentials']['usernames'].get(username, {})
        if not user_data:
            return None
            
        return {
            'username': username,
            'name': user_data.get('name', username),
            'email': user_data.get('email', ''),
            'role': user_data.get('role', 'viewer')
        }

authenticator = Authenticator()
__all__ = ['authenticator']
