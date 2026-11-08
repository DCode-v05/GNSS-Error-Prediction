"""Configuration settings for the ISRO GNSS Dashboard."""
from pathlib import Path
import os

BASE_DIR = Path(__file__).parent.parent.parent

class APIConfig:
    BASE_URL = os.getenv("API_URL", "http://localhost:8000")
    TIMEOUT = 30
    ENDPOINTS = {
        "health": "/api/v1/health",
        "predict": "/api/v1/predict",
        "satellites": "/api/v1/satellites",
        "telemetry": "/api/v1/telemetry",
        "alerts": "/api/v1/alerts"
    }

class DashboardConfig:
    TITLE = "ISRO GNSS Monitoring Dashboard"
    DESCRIPTION = """
    Real-time monitoring and analysis of GNSS satellite data and error predictions.
    Developed for ISRO's satellite operations team.
    """
    THEME = {
        "primaryColor": "#1E3D8F",
        "backgroundColor": "#F5F8FF",
        "secondaryBackgroundColor": "#E8F0FE",
        "textColor": "#1E3D8F",
        "font": "sans serif"
    }
    REFRESH_INTERVAL = 300000

class SatelliteConfig:
    CONSTELLATIONS = {
        "NAVIC": {"prefix": "IRNSS", "color": "#FF6B6B"},
        "GPS": {"prefix": "G", "color": "#4CC9F0"},
        "GLONASS": {"prefix": "R", "color": "#7209B7"},
        "GALILEO": {"prefix": "E", "color": "#3A86FF"},
        "BEIDOU": {"prefix": "C", "color": "#FFBE0B"}
    }
    DEFAULT_SATELLITES = ["IRNSS-1A", "IRNSS-1B", "IRNSS-1C", "IRNSS-1D", "G01", "G02", "R01", "R02"]

class AlertConfig:
    SEVERITY_LEVELS = {
        "CRITICAL": {"color": "#FF0000", "icon": "🚨"},
        "HIGH": {"color": "#FF6B6B", "icon": "⚠️"},
        "MEDIUM": {"color": "#FFD166", "icon": "ℹ️"},
        "LOW": {"color": "#06D6A0", "icon": "ℹ️"}
    }
    DEFAULT_ALERT_WINDOW = 24
