import requests
import json
from datetime import datetime, timedelta
import unittest
import traceback
import pprint

# Base URL for your Flask application
BASE_URL = 'http://localhost:5000/api'

class UrbanPlannerTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test suite configuration"""
        cls.base_url = BASE_URL
        cls.test_location_id = 6  # Consistent test location
        cls.test_planning_goals = ['sustainability', 'mobility', 'health']
        cls.pp = pprint.PrettyPrinter(indent=2)

    def _print_section_header(self, title):
        """Helper method to print formatted section headers"""
        print("\n" + "=" * 50)
        print(f"🔍 {title}")
        print("=" * 50)

    def test_detailed_traffic_prediction(self):
        """Detailed traffic prediction with comprehensive output"""
        self._print_section_header("Detailed Traffic Prediction Test")
        
        test_scenarios = [
            {
                'location_id': self.test_location_id,
                'timestamp': datetime.now().isoformat(),
                'day_of_week': datetime.now().weekday(),
                'hour_of_day': datetime.now().hour,
                'road_type': 'main_street'
            }
        ]

        for scenario in test_scenarios:
            response = requests.post(f'{self.base_url}/traffic/predict', json=scenario)
            self.assertEqual(response.status_code, 200, "Traffic prediction failed")
            
            prediction = response.json()
            
            print("\n🚗 Traffic Prediction Details:")
            print(f"Location ID: {scenario['location_id']}")
            print(f"Timestamp: {scenario['timestamp']}")
            print(f"Congestion Prediction:")
            self.pp.pprint({
                'Predicted Congestion': prediction['predicted_congestion'],
                'Congestion Category': prediction['congestion_level_category'],
                'Expected Vehicle Count': prediction['expected_vehicle_count']
            })
            
            print("\n📋 Traffic Recommendations:")
            for rec in prediction.get('recommendations', []):
                print(f"  - {rec}")
            
            # Validate prediction details
            self.assertTrue(0 <= prediction['predicted_congestion'] <= 1, "Invalid congestion prediction")
            self.assertIsInstance(prediction['recommendations'], list, "Recommendations must be a list")

    def test_detailed_environmental_prediction(self):
        """Detailed environmental prediction with comprehensive output"""
        self._print_section_header("Detailed Environmental Prediction Test")
        
        test_scenarios = [
            {
                'location_id': self.test_location_id,
                'traffic_level': 0.5,  # Medium traffic
                'current_air_quality': 75  # Moderate air quality
            }
        ]

        for scenario in test_scenarios:
            response = requests.post(f'{self.base_url}/environmental/predict', json=scenario)
            self.assertEqual(response.status_code, 200, "Environmental prediction failed")
            
            prediction = response.json()
            
            print("\n🌿 Environmental Prediction Details:")
            print(f"Location ID: {scenario['location_id']}")
            print(f"Environmental Metrics:")
            self.pp.pprint({
                'Predicted Air Quality Index': prediction['predicted_air_quality_index'],
                'Air Quality Category': prediction['air_quality_category'],
                'Green Cover Impact': prediction['green_cover_impact']
            })
            
            print("\n🍃 Environmental Recommendations:")
            for rec in prediction.get('recommendations', []):
                print(f"  - {rec}")
            
            # Validate prediction details
            self.assertTrue(0 <= prediction['predicted_air_quality_index'] <= 500, "Invalid air quality index")
            self.assertIsInstance(prediction['recommendations'], list, "Recommendations must be a list")

    def test_detailed_urban_planning_recommendations(self):
        """Detailed urban planning recommendations with comprehensive output"""
        self._print_section_header("Detailed Urban Planning Recommendations Test")
        
        test_scenarios = [
            {
                'location_id': self.test_location_id,
                'planning_goals': self.test_planning_goals
            }
        ]

        for scenario in test_scenarios:
            response = requests.post(f'{self.base_url}/recommendations', json=scenario)
            self.assertEqual(response.status_code, 200, "Urban planning recommendations failed")
            
            recommendations = response.json()
            
            print("\n🏙️ Urban Planning Recommendation Details:")
            print(f"Location ID: {scenario['location_id']}")
            print(f"Planning Goals: {scenario['planning_goals']}")
            
            print("\n📊 Key Metrics:")
            self.pp.pprint(recommendations.get('key_metrics', {}))
            
            print("\n🔬 Traffic Insights:")
            self.pp.pprint(recommendations.get('traffic_insights', {}))
            
            print("\n🌍 Environmental Insights:")
            self.pp.pprint(recommendations.get('environmental_insights', {}))
            
            print("\n📋 Strategic Recommendations:")
            for rec in recommendations.get('strategic_recommendations', []):
                print(f"  Priority {rec['priority']}: {rec['recommendation']}")
            
            # Validate recommendation structure
            self.assertIn('strategic_recommendations', recommendations, "Missing strategic recommendations")
            self.assertIsInstance(recommendations['strategic_recommendations'], list, "Strategic recommendations must be a list")
            self.assertTrue(len(recommendations['strategic_recommendations']) > 0, "No strategic recommendations generated")

def run_tests():
    """Run the entire test suite with detailed reporting"""
    print("🚀 Urban Planner Comprehensive Test Suite")
    print("=======================================")
    
    try:
        # Discover and run tests
        suite = unittest.TestLoader().loadTestsFromTestCase(UrbanPlannerTestSuite)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Print comprehensive test report
        print("\n📊 Test Report:")
        print(f"Total Tests: {result.testsRun}")
        print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        # Detailed failure and error reporting
        if result.failures:
            print("\n❌ Test Failures:")
            for failure in result.failures:
                print(f"Test: {failure[0]}")
                print(f"Error: {failure[1]}")
        
        if result.errors:
            print("\n❌ Test Errors:")
            for error in result.errors:
                print(f"Test: {error[0]}")
                print(f"Error: {error[1]}")
        
        return result.wasSuccessful()
    
    except Exception as e:
        print(f"\n❌ Unexpected error during test execution: {e}")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)