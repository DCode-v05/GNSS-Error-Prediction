"""
Simple authentication module for the ISRO GNSS Dashboard.

This module provides basic user authentication without using cookies.
"""
import streamlit as st
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import os
import hashlib
import binascii

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
CONFIG_FILE = CONFIG_DIR / "users.yaml"

class SimpleAuthenticator:
    """Handles simple user authentication without cookies."""
    
    def __init__(self):
        """Initialize the authenticator and load user configuration."""
        self.config = {}
        self._load_config()
    
    def _hash_password(self, password, salt=None):
        """Hash a password for storing."""
        if salt is None:
            salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
        else:
            salt = salt.encode('ascii')
            
        pwdhash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), salt, 100000)
        pwdhash = binascii.hexlify(pwdhash)
        return (salt + pwdhash).decode('ascii')
    
    def _load_config(self):
        """Load user configuration from YAML file."""
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            
            if not CONFIG_FILE.exists():
                print("Config file not found, creating default...")
                self._create_default_config()
                return
            
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
                    loaded_config = yaml.load(file, Loader=SafeLoader) or {}
                
                if not isinstance(loaded_config, dict) or 'users' not in loaded_config:
                    print("Invalid config structure, recreating...")
                    raise ValueError("Invalid configuration structure")
                    
                self.config = loaded_config
                print("Successfully loaded config file")
                
            except (yaml.YAMLError, ValueError) as e:
                print(f"Error loading config: {e}, recreating...")
                self._create_default_config()
                
        except Exception as e:
            print(f"Unexpected error loading config: {e}")
            st.error(f"Error loading configuration: {str(e)}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create a default configuration with default users."""
        try:
            print("Creating default configuration...")
            
            self.config = {
                'users': {
                    'admin': {
                        'email': 'admin@isro.gov.in',
                        'name': 'Admin User',
                        'password': self._hash_password('admin123'),
                        'role': 'admin'
                    },
                    'operator': {
                        'email': 'operator@isro.gov.in',
                        'name': 'Operator User',
                        'password': self._hash_password('operator123'),
                        'role': 'operator'
                    },
                    'viewer': {
                        'email': 'viewer@isro.gov.in',
                        'name': 'Viewer User',
                        'password': self._hash_password('viewer123'),
                        'role': 'viewer'
                    }
                }
            }
            
            os.makedirs(CONFIG_DIR, exist_ok=True)
            
            with open(CONFIG_FILE, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False)
                print(f"Successfully created default config at {CONFIG_FILE}")
                
        except Exception as e:
            print(f"Error creating default config: {e}")
            self.config = {'users': {}}
    
    def login(self):
        """Handle user login."""
        try:
            if not hasattr(self, 'config') or 'users' not in self.config:
                st.error("Configuration error. Please contact administrator.")
                return False
                
            if 'authenticated' not in st.session_state:
                st.session_state.authenticated = False
                st.session_state.user = None
            
            if st.session_state.authenticated:
                return True
            
            login_container = st.container()
            
            with login_container:
                st.title("ISRO GNSS Dashboard")
                st.markdown("---")
                with st.form('login_form'):
                    st.subheader('Login')
                    username = st.text_input('Username').strip()
                    password = st.text_input('Password', type='password')
                    
                    if st.form_submit_button('Login', key='login_submit'):
                        if not username or not password:
                            st.error('Please enter both username and password')
                            return False
                        
                        user = self.config['users'].get(username)
                        if not user:
                            st.error('Invalid username or password')
                            return False
                        
                        try:
                            stored_password = user.get('password', '')
                            if not stored_password or len(stored_password) < 64:
                                st.error('Invalid password format. Please contact administrator.')
                                return False
                                
                            salt = stored_password[:64]
                            hashed_password = self._hash_password(password, salt)
                            
                            if hashed_password == stored_password:
                                st.session_state.authenticated = True
                                st.session_state.user = {
                                    'username': username,
                                    'name': user.get('name', username),
                                    'email': user.get('email', ''),
                                    'role': user.get('role', 'viewer')
                                }
                                st.rerun()
                            else:
                                st.error('Invalid username or password')
                                return False
                                
                        except Exception as e:
                            print(f"Login error: {e}")
                            st.error('An error occurred during login')
                            return False
            
            return False
            
        except Exception as e:
            print(f"Unexpected error in login: {e}")
            st.error('An unexpected error occurred. Please try again.')
            return False
    
    def logout(self):
        """Handle user logout."""
        if st.sidebar.button("Logout", key="logout_button"):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()
    
    def get_current_user(self):
        """Get the current authenticated user's information."""
        if not st.session_state.get('authenticated'):
            return None
        return st.session_state.get('user')

authenticator = SimpleAuthenticator()
