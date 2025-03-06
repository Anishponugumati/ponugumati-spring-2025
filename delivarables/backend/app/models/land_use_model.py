# backend/app/models/land_use_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from datetime import datetime
from app.data.database import get_land_use_data, get_environmental_data, get_traffic_data

# Global variables to store model
model = None
scaler = None
model_file = 'backend/app/models/saved/land_use_model.joblib'
scaler_file = 'backend/app/models/saved/land_use_scaler.joblib'

def train_land_use_model():
    """Train land use optimization model using dummy data"""
    print("Training land use optimization model...")
    
    # Get data from database
    land_data = get_land_use_data()
    env_data = get_environmental_data()
    traffic_data = get_traffic_data()
    
    land_df = pd.DataFrame(land_data)
    env_df = pd.DataFrame(env_data)
    traffic_df = pd.DataFrame(traffic_data)
    
    # Convert timestamp to datetime if it's string
    if env_df['timestamp'].dtype == 'object':
        env_df['timestamp'] = pd.to_datetime(env_df['timestamp'])
    if traffic_df['timestamp'].dtype == 'object':
        traffic_df['timestamp'] = pd.to_datetime(traffic_df['timestamp'])
    
    # Aggregate environmental data by location
    env_agg = env_df.groupby('location_id').agg({
        'air_quality_index': 'mean',
        'noise_level': 'mean',
        'green_cover_percent': 'mean',
        'water_quality_index': 'mean'
    }).reset_index()
    
    # Aggregate traffic data by location
    traffic_df['date'] = traffic_df['timestamp'].dt.date
    traffic_agg = traffic_df.groupby('location_id').agg({
        'congestion_level': 'mean',
        'vehicle_count': 'mean',
        'average_speed': 'mean'
    }).reset_index()
    
    # Merge datasets
    merged_df = pd.merge(land_df, env_agg, on='location_id', how='left')
    merged_df = pd.merge(merged_df, traffic_agg, on='location_id', how='left')
    
    # Fill NaN values with means
    for col in merged_df.columns:
        if merged_df[col].dtype in [np.float64, np.int64] and merged_df[col].isna().any():
            merged_df[col] = merged_df[col].fillna(merged_df[col].mean())
    
    # Feature engineering
    # Calculate distance from city center (location_id 13 is assumed to be center in our 5x5 grid)
    city_center_loc = 13
    merged_df['distance_from_center'] = np.sqrt(
        (merged_df['latitude'] - merged_df.loc[merged_df['location_id'] == city_center_loc, 'latitude'].values[0])**2 +
        (merged_df['longitude'] - merged_df.loc[merged_df['location_id'] == city_center_loc, 'longitude'].values[0])**2
    )
    
    # One-hot encode current_use
    use_dummies = pd.get_dummies(merged_df['current_use'], prefix='use')
    merged_df = pd.concat([merged_df, use_dummies], axis=1)
    
    # One-hot encode zoning_code
    zone_dummies = pd.get_dummies(merged_df['zoning_code'], prefix='zone')
    merged_df = pd.concat([merged_df, zone_dummies], axis=1)
    
    # Select features
    features = ['area_size', 'population_density', 'building_density', 'property_value',
                'air_quality_index', 'noise_level', 'green_cover_percent', 'water_quality_index',
                'congestion_level', 'vehicle_count', 'average_speed', 'distance_from_center']
    features += list(use_dummies.columns) + list(zone_dummies.columns)
    
    # Target is land use type
    X = merged_df[features]
    y = merged_df['current_use']  # We're predicting optimal land use similar to current use
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_scaled, y)
    
    # Create directory if it doesn't exist
    os.makedirs('backend/app/models/saved', exist_ok=True)
    
    # Save model and scaler
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    
    print("Land use optimization model trained and saved.")
    
    return model, scaler, list(use_dummies.columns), list(zone_dummies.columns)
