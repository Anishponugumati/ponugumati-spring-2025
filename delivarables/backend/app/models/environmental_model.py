# backend/app/models/environmental_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib
import os
from datetime import datetime
from app.data.database import get_environmental_data, get_traffic_data

# Global variables to store model
model = None
scaler = None
model_file = 'backend/app/models/saved/environmental_model.joblib'
scaler_file = 'backend/app/models/saved/environmental_scaler.joblib'

def train_environmental_model():
    """Train environmental impact prediction model using dummy data"""
    print("Training environmental impact model...")
    
    # Get data from database
    env_data = get_environmental_data()
    traffic_data = get_traffic_data()
    
    env_df = pd.DataFrame(env_data)
    traffic_df = pd.DataFrame(traffic_data)
    
    # Convert timestamp to datetime if it's string
    if env_df['timestamp'].dtype == 'object':
        env_df['timestamp'] = pd.to_datetime(env_df['timestamp'])
    if traffic_df['timestamp'].dtype == 'object':
        traffic_df['timestamp'] = pd.to_datetime(traffic_df['timestamp'])
    
    # Prepare to join traffic and environmental data
    # For simplicity, we'll aggregate traffic data by day
    traffic_df['date'] = traffic_df['timestamp'].dt.date
    traffic_daily = traffic_df.groupby(['location_id', 'date']).agg({
        'congestion_level': 'mean',
        'vehicle_count': 'mean'
    }).reset_index()
    
    env_df['date'] = env_df['timestamp'].dt.date
    
    # Join datasets on location_id and date
    merged_df = pd.merge(
        env_df,
        traffic_daily,
        on=['location_id', 'date'],
        how='inner'
    )
    
    # Feature engineering
    merged_df['month'] = merged_df['timestamp'].dt.month
    merged_df['is_summer'] = merged_df['month'].apply(lambda x: 1 if 6 <= x <= 8 else 0)
    
    # Calculate distance from city center (using location_id as a proxy in our dummy data)
    # Assuming location_id 5 is the city center
    city_center_loc = 5
    merged_df['distance_from_center'] = abs(merged_df['location_id'] - city_center_loc)
    
    # Select features
    features = ['congestion_level', 'vehicle_count', 'month', 'is_summer', 
                'distance_from_center', 'location_id']
    
    # Target is air quality index
    X = merged_df[features]
    y = merged_df['air_quality_index']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)
    
    # Create directory if it doesn't exist
    os.makedirs('backend/app/models/saved', exist_ok=True)
    
    # Save model and scaler
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    
    print("Environmental impact model trained and saved.")
    
    return model, scaler

def load_environmental_model():
    """Load trained environmental model"""
    global model, scaler
    
    # Check if model files exist, train if they don't
    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        model, scaler = train_environmental_model()
    else:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
    
    return model, scaler

def get_air_quality_category(aqi):
    """Convert numerical AQI to category"""
    if aqi < 50:
        return "Good"
    elif aqi < 100:
        return "Moderate"
    elif aqi < 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi < 200:
        return "Unhealthy"
    elif aqi < 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def predict_environmental_impact(input_data):
    """Predict environmental impact based on input parameters"""
    global model, scaler
    
    # Load model if not loaded
    if model is None or scaler is None:
        model, scaler = load_environmental_model()
    
    # Extract input parameters
    location_id = input_data['location_id']
    traffic_level = input_data['traffic_level']  # Congestion level (0-1)
    current_air_quality = input_data.get('current_air_quality', 50)  # Default to moderate if not provided
    
    # Get additional context data from database
    env_data = get_environmental_data({'location_id': location_id})
    
    # Extract current date/time
    now = datetime.now()
    month = now.month
    is_summer = 1 if 6 <= month <= 8 else 0
    
    # Calculate distance from city center (using location_id as a proxy in our dummy data)
    # Assuming location_id 5 is the city center
    city_center_loc = 5
    distance_from_center = abs(location_id - city_center_loc)
    
    # Estimate vehicle count based on traffic level
    vehicle_count = traffic_level * 500
    
    # Prepare features
    features = [traffic_level, vehicle_count, month, is_summer, distance_from_center, location_id]
    features_df = pd.DataFrame([features], columns=['congestion_level', 'vehicle_count', 'month', 
                                                   'is_summer', 'distance_from_center', 'location_id'])
    
    # Scale features
    features_scaled = scaler.transform(features_df)
    
    # Make prediction
    predicted_aqi = model.predict(features_scaled)[0]
    
    # Adjust prediction based on current air quality
    # This simulates a more realistic model that takes current conditions into account
    adjusted_aqi = (predicted_aqi * 0.7) + (current_air_quality * 0.3)
    
    # Calculate other environmental metrics based on the predicted AQI
    # These are simplified estimates for demonstration purposes
    noise_level = 40 + (traffic_level * 40) + np.random.uniform(-5, 5)
    green_cover_effect = -5 * traffic_level  # Traffic reduces effective green cover
    
    # Get historical data for this location
    avg_aqi = np.mean([d['air_quality_index'] for d in env_data]) if env_data else None
    
    # Calculate projected impact on local greenery
    green_cover_impact = "Minimal" if traffic_level < 0.3 else "Moderate" if traffic_level < 0.7 else "Significant"
    
    # Calculate intervention options based on predicted AQI
    intervention_effectiveness = {}
    
    if predicted_aqi > 100:  # Unhealthy
        intervention_effectiveness = {
            "traffic_reduction_10_percent": -5 - np.random.uniform(1, 3),
            "traffic_reduction_25_percent": -12 - np.random.uniform(3, 7),
            "increase_green_cover_5_percent": -3 - np.random.uniform(1, 2),
            "increase_public_transit": -7 - np.random.uniform(2, 5)
        }
    elif predicted_aqi > 50:  # Moderate
        intervention_effectiveness = {
            "traffic_reduction_10_percent": -2 - np.random.uniform(0.5, 1.5),
            "traffic_reduction_25_percent": -5 - np.random.uniform(1, 3),
            "increase_green_cover_5_percent": -1 - np.random.uniform(0.5, 1),
            "increase_public_transit": -3 - np.random.uniform(1, 2)
        }
    else:  # Good
        intervention_effectiveness = {
            "traffic_reduction_10_percent": -1 - np.random.uniform(0.1, 0.5),
            "traffic_reduction_25_percent": -2 - np.random.uniform(0.5, 1),
            "increase_green_cover_5_percent": -0.5 - np.random.uniform(0.1, 0.5),
            "increase_public_transit": -1 - np.random.uniform(0.2, 0.8)
        }
    
    result = {
        'location_id': location_id,
        'predicted_air_quality_index': float(adjusted_aqi),
        'air_quality_category': get_air_quality_category(adjusted_aqi),
        'predicted_noise_level': float(noise_level),
        'green_cover_impact': green_cover_impact,
        'historical_comparison': {
            'average_aqi': float(avg_aqi) if avg_aqi is not None else None,
            'percent_change': float(((adjusted_aqi - avg_aqi) / avg_aqi) * 100) if avg_aqi is not None else None
        },
        'intervention_effectiveness': intervention_effectiveness
    }
    
    # Add recommendations based on air quality
    if adjusted_aqi > 150:
        result['recommendations'] = [
            "Implement traffic restrictions in this area immediately",
            "Issue public health advisories for sensitive groups",
            "Increase public transport capacity to reduce private vehicle use",
            "Consider temporary road closures during peak pollution hours"
        ]
    elif adjusted_aqi > 100:
        result['recommendations'] = [
            "Monitor air quality closely and prepare for possible restrictions",
            "Encourage work-from-home options to reduce commuter traffic",
            "Implement traffic calming measures",
            "Increase green barriers along busy roads"
        ]
    elif adjusted_aqi > 50:
        result['recommendations'] = [
            "Continue monitoring air quality trends",
            "Plan for increased green space in future development",
            "Incentivize electric vehicle use in this area",
            "Consider dedicated bus and cycle lanes"
        ]
    else:
        result['recommendations'] = [
            "Maintain current environmental policies",
            "Continue regular monitoring",
            "Document successful strategies for replication in other areas"
        ]
    
    return result

# Initialize model when module is imported
try:
    load_environmental_model()
except Exception as e:
    print(f"Error loading environmental model: {e}")
    print("Model will be trained on first prediction request.")