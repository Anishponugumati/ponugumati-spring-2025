from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import logging
import webbrowser
from threading import Timer
from util.Feature_extraction import extract_all_features_from_polygon
from util.meta_model import rule_based_meta_model

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for frontend integration

@app.route('/')
def index():
    print("Rendering index.html template")
    return render_template('index.html')

@app.route('/predict-zoning', methods=['POST'])
def predict_zoning():
    """
    API endpoint to analyze a GeoJSON polygon and return zoning recommendations
    along with improvement suggestions and extracted features.
    """
    print("Received request to /predict-zoning")
    
    # Step 1: Validate input
    try:
        geojson = request.get_json()
        print(f"Received GeoJSON: {json.dumps(geojson)[:200]}...")  # Print first 200 chars
        
        # Extract coordinates from GeoJSON
        if geojson.get('type') == 'Feature':
            geometry = geojson.get('geometry', {})
        else:
            geometry = geojson
            
        if geometry.get('type') != 'Polygon':
            print(f"Invalid geometry type: {geometry.get('type')}")
            return jsonify({'error': 'Invalid geometry type. Expected Polygon.'}), 400
            
        polygon = geometry.get('coordinates', [])[0]  # First ring of coordinates
        
        if not polygon or not isinstance(polygon, list):
            print("Invalid or missing polygon coordinates")
            return jsonify({'error': 'Invalid or missing polygon coordinates'}), 400
            
        print(f"Extracted polygon with {len(polygon)} points")
    
    except Exception as e:
        print(f"Error parsing request data: {str(e)}")
        return jsonify({'error': f'Error parsing request data: {str(e)}'}), 400
    
    # Step 2: Extract features
    try:
        print("Starting feature extraction...")
        print(f"Polygon coordinates: {polygon}...")  # Print first 5 points for debuggin
        features = extract_all_features_from_polygon(polygon)
        print(f"Features extracted: {features}")
    except Exception as e:
        print(f"Feature extraction failed: {str(e)}")
        return jsonify({'error': f'Feature extraction failed: {str(e)}'}), 500
    
    # Step 3: Meta-model decision
    try:
        print("Running meta-model...")
        zoning, suggestions = rule_based_meta_model(features)
        print(f"Zoning recommendation: {zoning}")
        print(f"Improvement suggestions: {suggestions}")
    except Exception as e:
        print(f"Meta-model failed: {str(e)}")
        return jsonify({'error': f'Meta-model failed: {str(e)}'}), 500
    
    # Step 4: Prepare mock data for air pollution and traffic
    # In a real app, these would come from separate models
    mock_air_data = {
        'aqi': 75,
        'pollutants': {
            'pm25': 12.5,
            'pm10': 35.2,
            'o3': 0.045,
            'no2': 38.6,
            'co': 1.2,
            'so2': 8.3
        }
    }
    
    mock_traffic_data = {
        'congestion_level': 'Medium',
        'metrics': {
            'average_speed': 28.5,
            'peak_hour_congestion': 'High',
            'public_transport_access': 'Good',
            'pedestrian_connectivity': 'Medium'
        }
    }
    
    # Step 5: Return combined response
    response = {
        'zoning_recommendation': zoning,
        'improvement_suggestions': suggestions,
        'features': features,
        'air_pollution': mock_air_data,
        'traffic': mock_traffic_data
    }
    
    print("Returning complete response")
    return jsonify(response), 200

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1.0, open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=True)