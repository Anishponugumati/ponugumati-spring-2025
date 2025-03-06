# backend/app/data/database.py
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import json
from geoalchemy2 import Geometry
from sqlalchemy.orm import declarative_base


import random

# Define base
Base = declarative_base()

# Create database engine
# DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///urban_planner.db')
DATABASE_URL = os.environ.get(
    'DATABASE_URL', 
    'mysql+pymysql://root:lohithMYSQL1@localhost/urbanplannerDB'
)


engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as connection:
        print("✅ Successfully connected to MySQL database!")
except Exception as e:
    print("❌ Connection failed:", e)


Session = sessionmaker(bind=engine)

# Define models
class TrafficData(Base):
    __tablename__ = 'traffic_data'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    location_id = Column(Integer)
    latitude = Column(Float)
    longitude = Column(Float)
    congestion_level = Column(Float)  # 0-1 scale
    vehicle_count = Column(Integer)
    average_speed = Column(Float)
    road_type = Column(String(255))
    
class EnvironmentalData(Base):
    __tablename__ = 'environmental_data'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    location_id = Column(Integer)
    latitude = Column(Float)
    longitude = Column(Float)
    air_quality_index = Column(Float)
    noise_level = Column(Float)
    green_cover_percent = Column(Float)
    water_quality_index = Column(Float)
    
class LandUseData(Base):
    __tablename__ = 'land_use_data'
    
    id = Column(Integer, primary_key=True)
    location_id = Column(Integer)
    latitude = Column(Float)
    longitude = Column(Float)
    area_size = Column(Float)  # in sq meters
    current_use = Column(String(255))  # residential, commercial, industrial, recreational, etc.
    zoning_code = Column(String(50))
    population_density = Column(Float)
    building_density = Column(Float)
    property_value = Column(Float)

def generate_dummy_traffic_data():
    """Generate realistic dummy traffic data"""
    data = []
    
    # Create 10 locations
    locations = [(40.712776 + i*0.01, -74.005974 + i*0.01) for i in range(10)]
    location_ids = list(range(1, 11))
    road_types = ['highway', 'main_street', 'residential', 'avenue', 'boulevard']
    
    # Generate data for past 30 days, hourly
    start_date = datetime.now() - timedelta(days=30)
    for day in range(30):
        for hour in range(24):
            current_time = start_date + timedelta(days=day, hours=hour)
            
            # Traffic patterns based on time of day
            is_weekday = current_time.weekday() < 5
            is_rush_hour_morning = 7 <= hour <= 9 and is_weekday
            is_rush_hour_evening = 16 <= hour <= 18 and is_weekday
            
            for idx, (lat, lng) in enumerate(locations):
                location_id = location_ids[idx]
                road_type = random.choice(road_types)
                
                # Higher congestion during rush hours
                base_congestion = 0.2
                if is_rush_hour_morning or is_rush_hour_evening:
                    base_congestion = 0.7
                
                # Add some randomness
                congestion_level = min(1.0, max(0.0, base_congestion + random.uniform(-0.2, 0.2)))
                
                # Vehicle count correlates with congestion
                vehicle_count = int(congestion_level * 500 + random.randint(-50, 50))
                
                # Speed inversely correlates with congestion
                avg_speed = 60 - (congestion_level * 50) + random.uniform(-5, 5)
                
                data.append({
                    'timestamp': current_time,
                    'location_id': location_id,
                    'latitude': lat + random.uniform(-0.002, 0.002),
                    'longitude': lng + random.uniform(-0.002, 0.002),
                    'congestion_level': congestion_level,
                    'vehicle_count': vehicle_count,
                    'average_speed': max(5, avg_speed),
                    'road_type': road_type
                })
    
    return data

def generate_dummy_environmental_data():
    """Generate realistic dummy environmental data"""
    data = []
    
    # Create 10 locations (slightly offset from traffic locations for variety)
    locations = [(40.712776 + i*0.01 + 0.005, -74.005974 + i*0.01 + 0.005) for i in range(10)]
    location_ids = list(range(1, 11))
    
    # Generate data for past 30 days, daily
    start_date = datetime.now() - timedelta(days=30)
    for day in range(30):
        current_time = start_date + timedelta(days=day)
        
        # Seasonal factors (simplified)
        seasonal_factor = 0.5 + 0.5 * np.sin(np.pi * day / 15)  # cycles over 30 days
        
        for idx, (lat, lng) in enumerate(locations):
            location_id = location_ids[idx]
            
            # More pollution in dense areas (lower location_ids in our dummy data)
            density_factor = 1 - (idx / 10)
            
            # Base values with some randomness and factors
            air_quality = 50 + density_factor * 50 + seasonal_factor * 20 + random.uniform(-10, 10)
            noise_level = 40 + density_factor * 40 + random.uniform(-5, 15)
            green_cover = 70 - density_factor * 40 + random.uniform(-10, 10)
            water_quality = 80 - density_factor * 30 + seasonal_factor * 10 + random.uniform(-10, 10)
            
            data.append({
                'timestamp': current_time,
                'location_id': location_id,
                'latitude': lat,
                'longitude': lng,
                'air_quality_index': air_quality,
                'noise_level': noise_level,
                'green_cover_percent': max(5, min(95, green_cover)),
                'water_quality_index': max(10, min(100, water_quality))
            })
    
    return data

def generate_dummy_land_use_data():
    """Generate realistic dummy land use data"""
    data = []
    
    # Create a grid of locations (5x5)
    base_lat, base_lng = 40.712776, -74.005974
    location_id = 1
    
    land_use_types = ['residential', 'commercial', 'industrial', 'recreational', 'mixed_use']
    zoning_codes = ['R1', 'R2', 'C1', 'C2', 'I1', 'I2', 'M1', 'P1']
    
    for i in range(5):
        for j in range(5):
            lat = base_lat + i*0.01
            lng = base_lng + j*0.01
            
            # Central areas are more dense and valuable
            distance_from_center = ((i-2)**2 + (j-2)**2)**0.5
            centrality_factor = 1 - (distance_from_center / 3)
            
            # Determine land use based on location
            if distance_from_center < 1:
                current_use = 'commercial' if random.random() < 0.7 else 'mixed_use'
            elif distance_from_center < 2:
                current_use = random.choice(['residential', 'mixed_use', 'commercial'])
            else:
                current_use = random.choice(['residential', 'industrial', 'recreational'])
            
            # Select appropriate zoning code
            if current_use == 'residential':
                zoning_code = random.choice(['R1', 'R2'])
            elif current_use == 'commercial':
                zoning_code = random.choice(['C1', 'C2'])
            elif current_use == 'industrial':
                zoning_code = random.choice(['I1', 'I2'])
            elif current_use == 'mixed_use':
                zoning_code = 'M1'
            else:  # recreational
                zoning_code = 'P1'
            
            # Calculate metrics based on land use and centrality
            area_size = random.uniform(5000, 50000)  # 5,000 to 50,000 sq meters
            
            if current_use == 'residential':
                population_density = 50 + centrality_factor * 150 + random.uniform(-20, 20)
                building_density = 0.3 + centrality_factor * 0.5 + random.uniform(-0.1, 0.1)
            elif current_use == 'commercial':
                population_density = 10 + centrality_factor * 40 + random.uniform(-5, 15)
                building_density = 0.5 + centrality_factor * 0.4 + random.uniform(-0.1, 0.1)
            elif current_use == 'industrial':
                population_density = 5 + random.uniform(-2, 10)
                building_density = 0.4 + random.uniform(-0.1, 0.2)
            elif current_use == 'mixed_use':
                population_density = 40 + centrality_factor * 100 + random.uniform(-15, 25)
                building_density = 0.4 + centrality_factor * 0.4 + random.uniform(-0.1, 0.1)
            else:  # recreational
                population_density = 2 + random.uniform(-1, 5)
                building_density = 0.05 + random.uniform(-0.02, 0.05)
            
            # Property value correlates with centrality and land use
            base_value = 1000 + centrality_factor * 4000
            if current_use == 'commercial':
                property_value = base_value * 1.5 + random.uniform(-500, 500)
            elif current_use == 'residential':
                property_value = base_value * 1.2 + random.uniform(-300, 300)
            elif current_use == 'mixed_use':
                property_value = base_value * 1.3 + random.uniform(-400, 400)
            elif current_use == 'industrial':
                property_value = base_value * 0.8 + random.uniform(-200, 200)
            else:  # recreational
                property_value = base_value * 0.6 + random.uniform(-100, 100)
            
            data.append({
                'location_id': location_id,
                'latitude': lat,
                'longitude': lng,
                'area_size': area_size,
                'current_use': current_use,
                'zoning_code': zoning_code,
                'population_density': max(1, population_density),
                'building_density': max(0.01, min(0.95, building_density)),
                'property_value': max(200, property_value)
            })
            
            location_id += 1
    
    return data

def save_dummy_data_to_csv():
    """Save dummy data to CSV files for easy loading"""
    # Create data directory if it doesn't exist
    os.makedirs('backend/app/data/dummy', exist_ok=True)
    
    # Generate data
    traffic_data = generate_dummy_traffic_data()
    environmental_data = generate_dummy_environmental_data()
    land_use_data = generate_dummy_land_use_data()
    
    # Save to CSV
    pd.DataFrame(traffic_data).to_csv('backend/app/data/dummy/traffic_data.csv', index=False)
    pd.DataFrame(environmental_data).to_csv('backend/app/data/dummy/environmental_data.csv', index=False)
    pd.DataFrame(land_use_data).to_csv('backend/app/data/dummy/land_use_data.csv', index=False)
    
    return {
        'traffic_data': len(traffic_data),
        'environmental_data': len(environmental_data),
        'land_use_data': len(land_use_data)
    }

def load_dummy_data_to_db():
    """Load dummy data from CSV files to database"""
    session = Session()
    
    # Clear existing data
    session.query(TrafficData).delete()
    session.query(EnvironmentalData).delete()
    session.query(LandUseData).delete()
    
    # Load from CSV
    traffic_df = pd.read_csv('backend/app/data/dummy/traffic_data.csv')
    env_df = pd.read_csv('backend/app/data/dummy/environmental_data.csv')
    land_df = pd.read_csv('backend/app/data/dummy/land_use_data.csv')
    
    # Convert timestamp strings to datetime objects
    traffic_df['timestamp'] = pd.to_datetime(traffic_df['timestamp'])
    env_df['timestamp'] = pd.to_datetime(env_df['timestamp'])
    
    # Add to database
    for _, row in traffic_df.iterrows():
        session.add(TrafficData(**row.to_dict()))
    
    for _, row in env_df.iterrows():
        session.add(EnvironmentalData(**row.to_dict()))
    
    for _, row in land_df.iterrows():
        session.add(LandUseData(**row.to_dict()))
    
    session.commit()
    session.close()
    
    return {
        'traffic_data': len(traffic_df),
        'environmental_data': len(env_df),
        'land_use_data': len(land_df)
    }

def init_db():
    """Initialize database, create tables and load dummy data"""
    # Create tables
    Base.metadata.create_all(engine)
    
    # Check if dummy data exists, generate if not
    if not os.path.exists('backend/app/data/dummy/traffic_data.csv'):
        print("Generating dummy data...")
        data_counts = save_dummy_data_to_csv()
        print(f"Generated dummy data: {data_counts}")
    
    # Load data into database
    data_counts = load_dummy_data_to_db()
    print(f"Loaded dummy data into database: {data_counts}")
    
    print("✅ Database setup and data loading completed successfully!")
    
    return True


def get_traffic_data(filters=None):
    """Retrieve traffic data with optional filters"""
    session = Session()
    query = session.query(TrafficData)
    
    if filters:
        if 'start_date' in filters and filters['start_date']:
            query = query.filter(TrafficData.timestamp >= filters['start_date'])
        if 'end_date' in filters and filters['end_date']:
            query = query.filter(TrafficData.timestamp <= filters['end_date'])
        if 'location_id' in filters and filters['location_id']:
            query = query.filter(TrafficData.location_id == filters['location_id'])
    
    results = query.all()
    session.close()
    
    return [row.__dict__ for row in results]

def get_environmental_data(filters=None):
    """Retrieve environmental data with optional filters"""
    session = Session()
    query = session.query(EnvironmentalData)
    
    if filters:
        if 'start_date' in filters and filters['start_date']:
            query = query.filter(EnvironmentalData.timestamp >= filters['start_date'])
        if 'end_date' in filters and filters['end_date']:
            query = query.filter(EnvironmentalData.timestamp <= filters['end_date'])
        if 'location_id' in filters and filters['location_id']:
            query = query.filter(EnvironmentalData.location_id == filters['location_id'])
    
    results = query.all()
    session.close()
    
    return [row.__dict__ for row in results]

def get_land_use_data(filters=None):
    """Retrieve land use data with optional filters"""
    session = Session()
    query = session.query(LandUseData)
    
    if filters:
        if 'location_id' in filters and filters['location_id']:
            query = query.filter(LandUseData.location_id == filters['location_id'])
        if 'current_use' in filters and filters['current_use']:
            query = query.filter(LandUseData.current_use == filters['current_use'])
    
    results = query.all()
    session.close()
    
    return [row.__dict__ for row in results]