"""Satellite visualization component for the ISRO GNSS Dashboard."""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

class SatelliteVisualization:
    def __init__(self):
        self.orbit_radius = 1.5
        self.earth_radius = 0.3
        
    def _create_earth(self):
        """Create a 3D sphere representing Earth."""
        phi = np.linspace(0, 2 * np.pi, 100)
        theta = np.linspace(0, np.pi, 100)
        
        x = self.earth_radius * np.outer(np.cos(phi), np.sin(theta))
        y = self.earth_radius * np.outer(np.sin(phi), np.sin(theta))
        z = self.earth_radius * np.outer(np.ones(np.size(phi)), np.cos(theta))
        
        return x, y, z
    
    def _create_orbit(self, altitude, inclination=45, phase=0):
        """Create a 3D orbit.
        
        Args:
            altitude (float): Altitude above Earth's surface (normalized)
            inclination (float): Orbit inclination in degrees
            phase (float): Orbit phase in radians
        """
        t = np.linspace(0, 2 * np.pi, 100)
        r = self.earth_radius + altitude
        
        inc = np.radians(inclination)
        
        x = r * np.cos(t + phase)
        y = r * np.sin(t + phase) * np.cos(inc)
        z = r * np.sin(t + phase) * np.sin(inc)
        
        return x, y, z
    
    def plot_satellite_positions(self, satellites):
        """Plot 3D visualization of satellite positions.
        
        Args:
            satellites (list): List of satellite dictionaries with position data
        """
        if not satellites:
            st.warning("No satellite data available")
            return
            
        fig = go.Figure()
        
        x, y, z = self._create_earth()
        fig.add_surface(x=x, y=y, z=z, 
                       colorscale=[[0, 'lightblue'], [1, 'blue']],
                       showscale=False)
        
        for sat in satellites:
            x_orb, y_orb, z_orb = self._create_orbit(
                altitude=sat.get('altitude', 0.5),
                inclination=sat.get('inclination', 45),
                phase=sat.get('phase', 0)
            )
            
            fig.add_trace(go.Scatter3d(
                x=x_orb, y=y_orb, z=z_orb,
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                name=f"{sat['id']} Orbit",
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[sat.get('x', 0)],
                y=[sat.get('y', 0)],
                z=[sat.get('z', 0)],
                mode='markers+text',
                marker=dict(
                    size=8,
                    color=sat.get('status_color', 'red'),
                    symbol='diamond'
                ),
                text=sat['id'],
                textposition='top center',
                name=sat['id'],
                showlegend=True
            ))
        
        fig.update_layout(
            title='Satellite Orbits',
            scene=dict(
                xaxis=dict(showbackground=False, title=''),
                yaxis=dict(showbackground=False, title=''),
                zaxis=dict(showbackground=False, title=''),
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_ground_track(self, ground_tracks):
        """Plot 2D ground track of satellites.
        
        Args:
            ground_tracks (dict): Dictionary mapping satellite IDs to lists of (lat, lon) points
        """
        if not ground_tracks:
            st.warning("No ground track data available")
            return
            
        fig = go.Figure()
        
        for sat_id, track in ground_tracks.items():
            if not track:
                continue
                
            lats, lons = zip(*track)
            
            fig.add_trace(go.Scattergeo(
                lon=lons,
                lat=lats,
                mode='lines',
                line=dict(width=2, color='red'),
                name=sat_id,
                hoverinfo='text',
                text=f"Satellite: {sat_id}"
            ))
        
        fig.update_geos(
            projection_type="orthographic",
            showocean=True, oceancolor="LightBlue",
            showland=True, landcolor="lightgray",
            showcountries=True, countrycolor="gray",
            showlakes=True, lakecolor="LightBlue",
            showcoastlines=True, coastlinecolor="gray"
        )
        
        fig.update_layout(
            title='Satellite Ground Tracks',
            geo=dict(
                showframe=False,
                projection_scale=1.5,
                center=dict(lat=20, lon=78),
                lonaxis=dict(showgrid=True, gridwidth=0.5, dtick=30),
                lataxis=dict(showgrid=True, gridwidth=0.5, dtick=20)
            ),
            height=500,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
