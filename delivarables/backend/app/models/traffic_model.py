import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from datetime import datetime, timedelta
from app.data.database import get_traffic_data

# Global variables to store model
model = None
scaler = None
model_file = 'backend/app/models/saved/traffic_model.joblib'
scaler_file = 'backend/app/models/saved/traffic_scaler.joblib'

def train_traffic_model():
    """Train traffic prediction model using dummy data"""
    print("Training traffic prediction model...")
    
    # Get data from database
    all_data = get_traffic_data()
    df = pd.DataFrame(all_data)
    
    # Convert timestamp to datetime if it's string
    if df['timestamp'].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Feature engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                          (df['hour'] >= 16) & (df['hour'] <= 18)).astype(int)
    
    # One-hot encode road_type
    road_type_dummies = pd.get_dummies(df['road_type'], prefix='road')
    df = pd.concat([df, road_type_dummies], axis=1)
    
    # Select features
    features = ['hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 
                'location_id'] + list(road_type_dummies.columns)
    
    X = df[features]
    y = df['congestion_level']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # Create directory if it doesn't exist
    os.makedirs('backend/app/models/saved', exist_ok=True)
    
    # Save model and scaler
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    
    print("Traffic prediction model trained and saved.")
    
    return model, scaler

def load_traffic_model():
    """Load trained traffic model"""
    global model, scaler
    
    # Check if model files exist, train if they don't
    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        model, scaler = train_traffic_model()
    else:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
    
    return model, scaler

def get_congestion_category(congestion_level):
    """Convert numerical congestion level to category"""
    if congestion_level < 0.3:
        return "Low"
    elif congestion_level < 0.6:
        return "Medium"
    else:
        return "High"

def predict_traffic(input_data):
    """Predict traffic congestion based on input parameters"""
    global model, scaler
    
    # Load model if not loaded
    if model is None or scaler is None:
        model, scaler = load_traffic_model()
    
    # Extract and prepare features
    timestamp = input_data['timestamp']
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)
    
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    is_rush_hour = 1 if ((hour >= 7 and hour <= 9) or (hour >= 16 and hour <= 18)) else 0
    location_id = input_data['location_id']
    
    # Get road type, default to main_street if not provided
    road_type = input_data.get('road_type', 'main_street')
    
    # Get all data to determine road types for one-hot encoding
    all_data = get_traffic_data()
    df_all = pd.DataFrame(all_data)
    road_types = sorted(df_all['road_type'].unique())  # Sort to ensure consistent order
    
    # Create one-hot encoded road type features
    road_type_features = [1 if road_type == rt else 0 for rt in road_types]
    
    # Create features with EXACT same column names and order as training
    features = [
        hour, 
        day_of_week, 
        is_weekend, 
        is_rush_hour, 
        location_id
    ] + road_type_features
    
    # Use predefined column names from training
    feature_columns = [
        'hour', 
        'day_of_week', 
        'is_weekend', 
        'is_rush_hour', 
        'location_id'
    ] + [f'road_{rt}' for rt in road_types]
    
    features_df = pd.DataFrame([features], columns=feature_columns)
    
    # Scale features
    features_scaled = scaler.transform(features_df)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    
    # Get historical data for context
    similar_conditions = df_all[
        (df_all['location_id'] == location_id) & 
        (df_all['timestamp'].dt.hour == hour) &
        (df_all['timestamp'].dt.dayofweek == day_of_week)
    ]
    
    avg_historical = similar_conditions['congestion_level'].mean() if not similar_conditions.empty else None
    max_historical = similar_conditions['congestion_level'].max() if not similar_conditions.empty else None
    
    # Calculate expected vehicle count and average speed based on congestion
    expected_vehicle_count = int(prediction * 500 + np.random.randint(-50, 50))
    expected_average_speed = 60 - (prediction * 50) + np.random.uniform(-5, 5)
    
    # Calculate percent change from historical average
    percent_change = None
    if avg_historical is not None and avg_historical > 0:
        percent_change = ((prediction - avg_historical) / avg_historical) * 100
    
    # Define result before returning
    result = {
        'predicted_congestion': float(prediction),
        'congestion_level_category': get_congestion_category(prediction),
        'expected_vehicle_count': expected_vehicle_count,
        'expected_average_speed': max(5, float(expected_average_speed)),
        'historical_comparison': {
            'average_congestion': float(avg_historical) if avg_historical is not None else None,
            'max_congestion': float(max_historical) if max_historical is not None else None,
            'percent_change': float(percent_change) if percent_change is not None else None
        },
        'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else timestamp,
        'location_id': location_id
    }
    
    # Add recommendations based on congestion level
    if prediction > 0.7:
        result['recommendations'] = [
            "Consider implementing congestion charges during peak hours",
            "Increase public transport capacity in this area",
            "Suggest alternative routes through city notification systems"
        ]
    elif prediction > 0.4:
        result['recommendations'] = [
            "Monitor traffic patterns for potential intervention",
            "Consider traffic light timing adjustments",
            "Prepare for potential congestion increases"
        ]
    else:
        result['recommendations'] = [
            "No immediate action needed",
            "Continue regular traffic monitoring"
        ]
    
    return result

# Initialize model when module is imported
try:
    load_traffic_model()
except Exception as e:
    print(f"Error loading traffic model: {e}")
    print("Model will be trained on first prediction request.")