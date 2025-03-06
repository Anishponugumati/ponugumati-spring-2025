import numpy as np
from datetime import datetime
from app.models.traffic_model import predict_traffic
from app.models.environmental_model import predict_environmental_impact
from app.models.land_use_model import train_land_use_model

def get_urban_planning_recommendations(input_data):
    """
    Generate comprehensive urban planning recommendations by integrating 
    predictions from traffic, environmental, and land use models.
    
    Args:
        input_data (dict): A dictionary containing:
            - location_id: int, identifying the specific urban area
            - planning_goals (list): List of city planning objectives
    
    Returns:
        dict: Comprehensive urban planning recommendations
    """
    location_id = input_data['location_id']
    planning_goals = input_data.get('planning_goals', [])
    
    # Ensure timestamp is set, default to current time if not provided
    timestamp = input_data.get('timestamp', datetime.now())
    if timestamp is None:
        timestamp = datetime.now()
    
    # Predict traffic conditions
    traffic_prediction = predict_traffic({
        'location_id': location_id,
        'timestamp': timestamp,
        'road_type': input_data.get('road_type', 'main_street')
    })
    
    # Predict environmental impact
    environmental_prediction = predict_environmental_impact({
        'location_id': location_id,
        'traffic_level': traffic_prediction['predicted_congestion'],
        'current_air_quality': input_data.get('current_air_quality', 50)
    })
    
    # Prepare recommendation framework
    recommendations = {
        'traffic_insights': traffic_prediction,
        'environmental_insights': environmental_prediction,
        'strategic_recommendations': [],
        'key_metrics': {
            'traffic_congestion': traffic_prediction['predicted_congestion'],
            'air_quality_index': environmental_prediction['predicted_air_quality_index'],
            'green_cover_impact': environmental_prediction['green_cover_impact']
        }
    }
    
    # Prioritize recommendations based on planning goals
    def add_strategic_recommendation(recommendation, priority=1):
        recommendations['strategic_recommendations'].append({
            'recommendation': recommendation,
            'priority': priority
        })
    
    # Traffic-based recommendations
    if traffic_prediction['predicted_congestion'] > 0.6:
        add_strategic_recommendation(
            "Develop comprehensive traffic management strategy to reduce congestion", 
            priority=3
        )
        if "sustainability" in planning_goals:
            add_strategic_recommendation(
                "Invest in public transportation and bike infrastructure", 
                priority=2
            )
    
    # Environmental recommendations
    if environmental_prediction['predicted_air_quality_index'] > 100:
        add_strategic_recommendation(
            "Implement green infrastructure and emission reduction programs", 
            priority=3
        )
        if "health" in planning_goals:
            add_strategic_recommendation(
                "Create more green spaces and urban parks to improve air quality", 
                priority=2
            )
    
    # Specific goal-based recommendations
    if "mobility" in planning_goals:
        add_strategic_recommendation(
            "Optimize public transit routes and increase frequency during peak hours", 
            priority=1
        )
    
    if "sustainability" in planning_goals:
        add_strategic_recommendation(
            "Develop incentive programs for electric and low-emission vehicles", 
            priority=1
        )
    
    if "economic_development" in planning_goals:
        add_strategic_recommendation(
            "Create mixed-use development zones to reduce commute times", 
            priority=2
        )
    
    if "resilience" in planning_goals:
        add_strategic_recommendation(
            "Design adaptable urban infrastructure with climate change considerations", 
            priority=1
        )
    
    # Optional land use optimization recommendation
    try:
        # Attempt to train land use model for additional insights
        _, _, use_features, zone_features = train_land_use_model()
        recommendations['land_use_optimization_features'] = {
            'use_categories': use_features,
            'zoning_categories': zone_features
        }
        
        if "urban_design" in planning_goals:
            add_strategic_recommendation(
                "Conduct detailed land use optimization study for zoning improvements", 
                priority=2
            )
    except Exception as e:
        print(f"Land use model training failed: {e}")
    
    # Sort recommendations by priority
    recommendations['strategic_recommendations'].sort(key=lambda x: x['priority'])
    
    return recommendations