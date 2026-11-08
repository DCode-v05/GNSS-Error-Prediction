"""
Alerts Page

This module provides the UI and functionality for managing alerts in the ISRO GNSS Dashboard.
"""
import streamlit as st
from datetime import datetime, timedelta

class AlertsPage:
    """Handles the display of alerts and notifications."""
    
    def __init__(self):
        """Initialize the AlertsPage with sample data."""
        self.alerts = self._load_sample_alerts()
    
    def _load_sample_alerts(self):
        """Load sample alert data."""
        return [
            {
                'id': 'ALT-001',
                'title': 'Signal Degradation Detected',
                'message': 'Signal strength below threshold for IRNSS-1C',
                'severity': 'High',
                'status': 'Open',
                'timestamp': (datetime.now() - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S'),
                'assigned_to': 'Ground Control',
                'satellite': 'IRNSS-1C',
                'source': 'Automated Monitoring'
            },
            {
                'id': 'ALT-002',
                'title': 'Scheduled Maintenance',
                'message': 'Planned maintenance for IRNSS-1B',
                'severity': 'Medium',
                'status': 'Acknowledged',
                'timestamp': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
                'assigned_to': 'Maintenance Team',
                'satellite': 'IRNSS-1B',
                'source': 'Scheduler'
            },
            {
                'id': 'ALT-003',
                'title': 'Orbit Adjustment Required',
                'message': 'Minor orbit correction needed for IRNSS-1A',
                'severity': 'Medium',
                'status': 'In Progress',
                'timestamp': (datetime.now() - timedelta(hours=5)).strftime('%Y-%m-%d %H:%M:%S'),
                'assigned_to': 'Orbit Control',
                'satellite': 'IRNSS-1A',
                'source': 'Orbit Analysis'
            },
            {
                'id': 'ALT-004',
                'title': 'Battery Health Warning',
                'message': 'Battery capacity below 80% on IRNSS-1D',
                'severity': 'High',
                'status': 'Open',
                'timestamp': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'),
                'assigned_to': 'Power Systems',
                'satellite': 'IRNSS-1D',
                'source': 'Telemetry'
            }
        ]
    
    def display(self):
        """Display the alerts interface."""
        st.title("Alerts & Notifications")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_filter = st.selectbox(
                "Status",
                ["All", "Open", "Acknowledged", "In Progress", "Resolved"]
            )
        
        with col2:
            severity_filter = st.selectbox(
                "Severity",
                ["All", "Critical", "High", "Medium", "Low"]
            )
        
        with col3:
            assigned_filter = st.selectbox(
                "Assigned To",
                ["All", "Ground Control", "Maintenance Team", "Orbit Control", "Power Systems"]
            )
        
        filtered_alerts = self.alerts
        
        if status_filter != "All":
            filtered_alerts = [alert for alert in filtered_alerts 
                             if alert['status'] == status_filter]
        
        if severity_filter != "All":
            filtered_alerts = [alert for alert in filtered_alerts 
                             if alert['severity'] == severity_filter]
        
        if assigned_filter != "All":
            filtered_alerts = [alert for alert in filtered_alerts 
                             if alert['assigned_to'] == assigned_filter]
        
        st.subheader(f"Active Alerts ({len(filtered_alerts)})")
        
        for alert in filtered_alerts:
            self._display_alert_card(alert)
        
        user_role = st.session_state.get('user', {}).get('role', 'viewer')
        if user_role in ['admin', 'operator']:
            if st.button("Create New Alert"):
                st.session_state['show_new_alert'] = True
            
            if st.session_state.get('show_new_alert', False):
                self._show_new_alert_form()
    
    def _display_alert_card(self, alert):
        """Display a single alert as a card."""
        if alert['severity'] == 'Critical':
            border_color = "#ff4444"
        elif alert['severity'] == 'High':
            border_color = "#ffbb33"
        elif alert['severity'] == 'Medium':
            border_color = "#ffcc00"
        else:
            border_color = "#4285F4"
        
        st.markdown(
            f"""
            <div style="
                border-left: 5px solid {border_color};
                background-color: #f8f9fa;
                padding: 10px 15px;
                margin: 10px 0;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0;">{alert['title']}</h4>
                    <span style="font-size: 0.8em; color: #666;">{alert['timestamp']}</span>
                </div>
                <p style="margin: 5px 0 10px 0;">{alert['message']}</p>
                <div style="display: flex; justify-content: space-between; font-size: 0.9em;">
                    <div>
                        <span style="font-weight: bold;">Severity:</span> {alert['severity']} | 
                        <span style="font-weight: bold;">Status:</span> {alert['status']} |
                        <span style="font-weight: bold;">Satellite:</span> {alert['satellite']}
                    </div>
                    <div>
                        <span style="font-weight: bold;">Assigned to:</span> {alert['assigned_to']}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col1, col2, col3, _ = st.columns([1, 1, 1, 5])
        
        with col1:
            if st.button(f"Acknowledge {alert['id']}", key=f"ack_{alert['id']}"):
                alert['status'] = 'Acknowledged'
                st.rerun()
        
        with col2:
            if st.button(f"Resolve {alert['id']}", key=f"resolve_{alert['id']}"):
                alert['status'] = 'Resolved'
                st.rerun()
        
        with col3:
            if st.button(f"Details {alert['id']}", key=f"details_{alert['id']}"):
                st.session_state['selected_alert'] = alert
                st.rerun()
        
        if st.session_state.get('selected_alert', {}).get('id') == alert['id']:
            self._show_alert_details(alert)
    
    def _show_alert_details(self, alert):
        """Show detailed information about an alert."""
        with st.expander("Alert Details", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Alert ID:** {alert['id']}")
                st.markdown(f"**Title:** {alert['title']}")
                st.markdown(f"**Severity:** {alert['severity']}")
                st.markdown(f"**Status:** {alert['status']}")
            
            with col2:
                st.markdown(f"**Satellite:** {alert['satellite']}")
                st.markdown(f"**Assigned To:** {alert['assigned_to']}")
                st.markdown(f"**Source:** {alert['source']}")
                st.markdown(f"**Timestamp:** {alert['timestamp']}")
            
            st.markdown("**Description:**")
            st.info(alert['message'])
            
            notes = st.text_area("Add Notes", key=f"notes_{alert['id']}")
            
            col1, col2 = st.columns([1, 5])
            
            with col1:
                if st.button("Save Notes", key=f"save_notes_{alert['id']}"):
                    if 'notes' not in alert:
                        alert['notes'] = []
                    alert['notes'].append({
                        'text': notes,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'user': st.session_state.get('user', {}).get('name', 'System')
                    })
                    st.success("Notes saved successfully!")
            
            if 'notes' in alert and alert['notes']:
                st.markdown("**Notes History:**")
                for note in reversed(alert['notes']):
                    st.markdown(
                        f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 5px 0;'>"
                        f"<div style='font-size: 0.8em; color: #666;'>{note['user']} at {note['timestamp']}</div>"
                        f"<div style='margin-top: 5px;'>{note['text']}</div>"
                        "</div>",
                        unsafe_allow_html=True
                    )
    
    def _show_new_alert_form(self):
        """Show form to create a new alert."""
        with st.form("new_alert"):
            st.subheader("Create New Alert")
            
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Title*")
                severity = st.selectbox(
                    "Severity*",
                    ["Critical", "High", "Medium", "Low"]
                )
                satellite = st.selectbox(
                    "Satellite*",
                    ["IRNSS-1A", "IRNSS-1B", "IRNSS-1C", "IRNSS-1D", "IRNSS-1E", "IRNSS-1F", "IRNSS-1G", "IRNSS-1I"]
                )
            
            with col2:
                assigned_to = st.selectbox(
                    "Assign To*",
                    ["Ground Control", "Maintenance Team", "Orbit Control", "Power Systems", "Operations"]
                )
                status = st.selectbox(
                    "Status*",
                    ["Open", "Acknowledged", "In Progress", "Resolved"]
                )
                source = st.selectbox(
                    "Source*",
                    ["Manual", "Automated Monitoring", "Telemetry", "Scheduler", "Other"]
                )
            
            message = st.text_area("Message*")
            
            if st.form_submit_button("Create Alert"):
                if title and message:
                    new_alert = {
                        'id': f"ALT-{len(self.alerts) + 100:03d}",
                        'title': title,
                        'message': message,
                        'severity': severity,
                        'status': status,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'assigned_to': assigned_to,
                        'satellite': satellite,
                        'source': source,
                        'notes': []
                    }
                    self.alerts.insert(0, new_alert)
                    st.session_state['show_new_alert'] = False
                    st.rerun()
                else:
                    st.error("Please fill in all required fields.")
            
            if st.form_submit_button("Cancel"):
                st.session_state['show_new_alert'] = False
                st.rerun()

alerts_page = AlertsPage()

def display():
    """Display the alerts page."""
    alerts_page.display()

if __name__ == "__main__":
    display()
