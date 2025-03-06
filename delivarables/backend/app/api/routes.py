# backend/app/api/routes.py
from flask import Blueprint, request, jsonify
from app.data.database import get_traffic_data, get_environmental_data, get_land_use_data
from app.models.traffic_model import predict_traffic
from app.models.environmental_model import predict_environmental_impact
from app.models.land_use_model import train_land_use_model
from app.models.meta_model import get_urban_planning_recommendations
from datetime import datetime

api_bp = Blueprint('api', __name__)

@api_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@api_bp.route('/traffic/data', methods=['GET'])
def traffic_data():
    """Get traffic data with optional filters"""
    filters = {}
    
    if 'start_date' in request.args:
        filters['start_date'] = datetime.fromisoformat(request.args.get('start_date'))
    if 'end_date' in request.args:
        filters['end_date'] = datetime.fromisoformat(request.args.get('end_date'))
    if 'location_id' in request.args:
        filters['location_id'] = int(request.args.get('location_id'))
    
    data = get_traffic_data(filters)
    
    # Convert SQLAlchemy objects to JSON
    for item in data:
        if '_sa_instance_state' in item:
            del item['_sa_instance_state']
        if isinstance(item.get('timestamp'), datetime):
            item['timestamp'] = item['timestamp'].isoformat()
    
    return jsonify(data)

@api_bp.route('/environmental/data', methods=['GET'])
def environmental_data():
    """Get environmental data with optional filters"""
    filters = {}
    
    if 'start_date' in request.args:
        filters['start_date'] = datetime.fromisoformat(request.args.get('start_date'))
    if 'end_date' in request.args:
        filters['end_date'] = datetime.fromisoformat(request.args.get('end_date'))
    if 'location_id' in request.args:
        filters['location_id'] = int(request.args.get('location_id'))
    
    data = get_environmental_data(filters)
    
    # Convert SQLAlchemy objects to JSON
    for item in data:
        if '_sa_instance_state' in item:
            del item['_sa_instance_state']
        if isinstance(item.get('timestamp'), datetime):
            item['timestamp'] = item['timestamp'].isoformat()
    
    return jsonify(data)

@api_bp.route('/land-use/data', methods=['GET'])
def land_use_data():
    """Get land use data with optional filters"""
    filters = {}
    
    if 'location_id' in request.args:
        filters['location_id'] = int(request.args.get('location_id'))
    if 'current_use' in request.args:
        filters['current_use'] = request.args.get('current_use')
    
    data = get_land_use_data(filters)
    
    # Convert SQLAlchemy objects to JSON
    for item in data:
        if '_sa_instance_state' in item:
            del item['_sa_instance_state']
    
    return jsonify(data)

@api_bp.route('/traffic/predict', methods=['POST'])
def predict_traffic_endpoint():
    """Predict traffic congestion based on input parameters"""
    data = request.json
    
    required_params = ['location_id', 'timestamp', 'day_of_week', 'hour_of_day']
    for param in required_params:
        if param not in data:
            return jsonify({'error': f'Missing required parameter: {param}'}), 400
    
    # Convert timestamp string to datetime if needed
    if isinstance(data['timestamp'], str):
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
    
    prediction = predict_traffic(data)
    return jsonify(prediction)

@api_bp.route('/environmental/predict', methods=['POST'])
def predict_environmental_endpoint():
    """Predict environmental impact based on input parameters"""
    data = request.json
    
    required_params = ['location_id', 'traffic_level', 'current_air_quality']
    for param in required_params:
        if param not in data:
            return jsonify({'error': f'Missing required parameter: {param}'}), 400
    
    prediction = predict_environmental_impact(data)
    return jsonify(prediction)

@api_bp.route('/land-use/optimize', methods=['POST'])
def optimize_land_use_endpoint():
    """Optimize land use based on input parameters"""
    data = request.json
    
    required_params = ['location_id', 'constraints']
    for param in required_params:
        if param not in data:
            return jsonify({'error': f'Missing required parameter: {param}'}), 400
    
    optimization = optimize_land_use(data)
    return jsonify(optimization)

@api_bp.route('/recommendations', methods=['POST'])
def get_recommendations_endpoint():
    """Get urban planning recommendations based on all models"""
    data = request.json
    
    required_params = ['location_id', 'planning_goals']
    for param in required_params:
        if param not in data:
            return jsonify({'error': f'Missing required parameter: {param}'}), 400
    
    recommendations = get_urban_planning_recommendations(data)
    return jsonify(recommendations)