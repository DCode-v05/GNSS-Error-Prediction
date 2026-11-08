"""
Admin Panel

This module provides administrative functionality for the ISRO GNSS Dashboard.
"""
import streamlit as st
import yaml
from pathlib import Path
import os

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
CONFIG_FILE = CONFIG_DIR / "users.yaml"

class AdminPanel:
    """Handles the admin functionality of the dashboard."""
    
    def __init__(self):
        """Initialize the admin panel."""
        self.config = self._load_config()
    
    def _load_config(self):
        """Load the user configuration."""
        try:
            with open(CONFIG_FILE, 'r') as file:
                return yaml.safe_load(file) or {'users': {}}
        except Exception as e:
            st.error(f"Failed to load configuration: {e}")
            return {'users': {}}
    
    def _save_config(self):
        """Save the user configuration."""
        try:
            with open(CONFIG_FILE, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            return True
        except Exception as e:
            st.error(f"Failed to save configuration: {e}")
            return False
    
    def display(self):
        """Display the admin panel."""
        st.title("Admin Panel")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "User Management", 
            "System Settings", 
            "Audit Logs", 
            "About"
        ])
        
        with tab1:
            self._display_user_management()
        
        with tab2:
            self._display_system_settings()
        
        with tab3:
            self._display_audit_logs()
        
        with tab4:
            self._display_about()
    
    def _display_user_management(self):
        """Display the user management interface."""
        st.header("User Management")
        
        with st.expander("Add New User", expanded=False):
            with st.form("add_user"):
                col1, col2 = st.columns(2)
                
                with col1:
                    username = st.text_input("Username*")
                    email = st.text_input("Email*")
                
                with col2:
                    full_name = st.text_input("Full Name*")
                    role = st.selectbox("Role*", ["admin", "operator", "viewer"])
                
                password = st.text_input("Temporary Password*", type="password")
                confirm_password = st.text_input("Confirm Password*", type="password")
                
                if st.form_submit_button("Add User"):
                    if not all([username, email, full_name, password, confirm_password]):
                        st.error("Please fill in all required fields.")
                    elif password != confirm_password:
                        st.error("Passwords do not match.")
                    elif username in self.config.get('users', {}):
                        st.error("Username already exists.")
                    else:
                        self.config.setdefault('users', {})[username] = {
                            'email': email,
                            'name': full_name,
                            'password': password,
                            'role': role,
                            'created_at': str(st.session_state.get('current_time', 'N/A')),
                            'created_by': st.session_state.get('user', {}).get('username', 'system')
                        }
                        if self._save_config():
                            st.success(f"User '{username}' added successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to save user.")
        
        st.subheader("Existing Users")
        users = self.config.get('users', {})
        
        if not users:
            st.info("No users found.")
            return
        
        user_data = []
        for username, user_info in users.items():
            user_data.append({
                'Username': username,
                'Name': user_info.get('name', 'N/A'),
                'Email': user_info.get('email', 'N/A'),
                'Role': user_info.get('role', 'viewer').capitalize(),
                'Created At': user_info.get('created_at', 'N/A'),
                'Created By': user_info.get('created_by', 'N/A'),
                'Actions': username
            })
        
        for user in user_data:
            col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 2, 3, 1, 2, 2, 3])
            
            with col1:
                st.write(user['Username'])
            
            with col2:
                st.write(user['Name'])
            
            with col3:
                st.write(user['Email'])
            
            with col4:
                st.write(user['Role'])
            
            with col5:
                st.write(user['Created At'])
            
            with col6:
                st.write(user['Created By'])
            
            with col7:
                current_user = st.session_state.get('user', {})
                if current_user.get('role') == 'admin' or current_user.get('username') == user['Username']:
                    edit_key = f"edit_{user['Username']}"
                    delete_key = f"delete_{user['Username']}"
                    
                    if st.button("Edit", key=edit_key):
                        st.session_state['edit_user'] = user['Username']
                    
                    if current_user.get('username') != user['Username']:
                        if st.button("Delete", key=delete_key):
                            if user['Username'] in self.config.get('users', {}):
                                del self.config['users'][user['Username']]
                                if self._save_config():
                                    st.success(f"User '{user['Username']}' deleted successfully!")
                                    st.rerun()
            
            if st.session_state.get('edit_user') == user['Username']:
                self._show_edit_user_form(user['Username'])
            
            st.markdown("---")
    
    def _show_edit_user_form(self, username):
        """Show form to edit a user."""
        user = self.config.get('users', {}).get(username, {})
        
        with st.form(f"edit_user_{username}"):
            st.subheader(f"Edit User: {username}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_email = st.text_input("Email*", value=user.get('email', ''))
                new_role = st.selectbox(
                    "Role*",
                    ["admin", "operator", "viewer"],
                    index=["admin", "operator", "viewer"].index(user.get('role', 'viewer'))
                )
            
            with col2:
                new_name = st.text_input("Full Name*", value=user.get('name', ''))
                new_password = st.text_input("New Password (leave blank to keep current)", type="password")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.form_submit_button("Save Changes"):
                    if not all([new_email, new_name]):
                        st.error("Please fill in all required fields.")
                    else:
                        self.config['users'][username].update({
                            'email': new_email,
                            'name': new_name,
                            'role': new_role,
                            'updated_at': str(st.session_state.get('current_time', 'N/A')),
                            'updated_by': st.session_state.get('user', {}).get('username', 'system')
                        })
                        
                        if new_password:
                            self.config['users'][username]['password'] = new_password
                        
                        if self._save_config():
                            st.success("User updated successfully!")
                            del st.session_state['edit_user']
                            st.rerun()
                        else:
                            st.error("Failed to update user.")
            
            with col2:
                if st.form_submit_button("Cancel"):
                    del st.session_state['edit_user']
                    st.rerun()
    
    def _display_system_settings(self):
        """Display system settings."""
        st.header("System Settings")
        
        st.subheader("System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Users", len(self.config.get('users', {})))
            st.metric("Active Sessions", "N/A")
        
        with col2:
            st.metric("System Uptime", "24d 6h 12m")
            st.metric("Database Status", "Connected")
        
        st.subheader("Configuration")
        
        with st.form("system_config"):
            st.checkbox("Enable Email Notifications", value=True, key="email_notifications")
            st.checkbox("Maintenance Mode", value=False, key="maintenance_mode")
            log_level = st.selectbox(
                "Log Level",
                ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                index=1
            )
            
            if st.form_submit_button("Save Settings"):
                st.success("System settings updated successfully!")
    
    def _display_audit_logs(self):
        """Display audit logs."""
        st.header("Audit Logs")
        
        sample_logs = [
            {"timestamp": "2023-05-15 10:30:45", "user": "admin", "action": "User login", "ip": "192.168.1.1"},
            {"timestamp": "2023-05-15 09:15:22", "user": "operator1", "action": "Updated satellite data", "ip": "192.168.1.2"},
            {"timestamp": "2023-05-14 16:45:10", "user": "admin", "action": "Added new user: operator2", "ip": "192.168.1.1"},
            {"timestamp": "2023-05-14 14:20:33", "user": "viewer1", "action": "Viewed dashboard", "ip": "192.168.1.3"},
            {"timestamp": "2023-05-13 11:05:17", "user": "admin", "action": "System backup completed", "ip": "192.168.1.1"},
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            user_filter = st.selectbox(
                "Filter by User",
                ["All Users"] + sorted(set(log["user"] for log in sample_logs))
            )
        
        with col2:
            date_filter = st.date_input("Filter by Date")
        
        filtered_logs = sample_logs
        if user_filter != "All Users":
            filtered_logs = [log for log in filtered_logs if log["user"] == user_filter]
        
        if date_filter:
            date_str = date_filter.strftime("%Y-%m-%d")
            filtered_logs = [log for log in filtered_logs if log["timestamp"].startswith(date_str)]
        
        if filtered_logs:
            st.table(filtered_logs)
        else:
            st.info("No logs found matching the selected filters.")
        
        if st.button("Export Logs"):
            st.success("Logs exported successfully!")
    
    def _display_about(self):
        """Display about information."""
        st.header("About")
        
        st.markdown("""
        ### ISRO GNSS Dashboard Admin Panel
        
        **Version:** 1.0.0  
        **Last Updated:** May 15, 2023
        
        ---
        
        This admin panel provides tools for managing users, system settings, and monitoring 
        the ISRO GNSS Dashboard.
        
        ### Features
        - **User Management**: Add, edit, and remove user accounts
        - **System Settings**: Configure application settings
        - **Audit Logs**: View system activity and user actions
        
        ### Support
        For assistance, please contact the system administrator or email support@isro.gov.in
        
        ---
        
        © 2023 ISRO - Indian Space Research Organisation
        """)

admin_panel = AdminPanel()

def display():
    """Display the admin panel."""
    if st.session_state.get('user', {}).get('role') != 'admin':
        st.error("You do not have permission to access this page.")
        return
    
    st.session_state['current_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    admin_panel.display()

if __name__ == "__main__":
    display()
