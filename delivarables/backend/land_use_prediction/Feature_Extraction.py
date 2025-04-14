import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import nearest_points

# Load the land-use dataset and highway dataset
land_use_gdf = gpd.read_file("osm_data/land_use_data.geojson")
highway_gdf = gpd.read_file("osm_data/highway_data.geojson")

# Define a projected CRS (e.g., EPSG:3857 for meters)
projected_crs = "EPSG:3857"

# Convert both GeoDataFrames to the projected CRS
land_use_proj = land_use_gdf.to_crs(projected_crs)
highway_proj = highway_gdf.to_crs(projected_crs)

# Compute area in square meters using the projected CRS
land_use_proj["area"] = land_use_proj.geometry.area

# Compute centroids in the projected CRS
land_use_proj["centroid"] = land_use_proj.geometry.centroid
land_use_proj["centroid_x_proj"] = land_use_proj.centroid.x
land_use_proj["centroid_y_proj"] = land_use_proj.centroid.y

# Convert the centroids back to geographic CRS for latitude and longitude
centroids_geo = land_use_proj["centroid"].to_crs("EPSG:4326")
land_use_proj["latitude"] = centroids_geo.y
land_use_proj["longitude"] = centroids_geo.x

# Now, for nearest highway distance, we can work in the projected CRS
def find_nearest_highway(polygon, highway_gdf_proj):
    if highway_gdf_proj.empty:
        return None
    nearest_geom = nearest_points(polygon.centroid, highway_gdf_proj.union_all())[1]
    return polygon.centroid.distance(nearest_geom)

land_use_proj["nearest_highway_distance"] = land_use_proj.geometry.apply(
    lambda geom: find_nearest_highway(geom, highway_proj)
)

# Create a buffer of 500 meters around each polygon for counting neighboring zones
buffered_land_use = land_use_proj.copy()
buffered_land_use["buffer"] = buffered_land_use.geometry.buffer(500)

def count_neighbors(polygon, land_use_df):
    return land_use_df[land_use_df.geometry.intersects(polygon)].shape[0]

land_use_proj["neighboring_zones"] = land_use_proj.geometry.apply(
    lambda geom: count_neighbors(geom, buffered_land_use)
)

# Compute shape complexity (perimeter/area ratio) in the projected CRS
land_use_proj["shape_complexity"] = land_use_proj.geometry.length / land_use_proj["area"]

# Select final features and convert back to geographic CRS for saving if needed
# (Here we keep latitude and longitude already computed)
final_features = land_use_proj[[
    "landuse", "area", "latitude", "longitude",
    "nearest_highway_distance", "neighboring_zones", "shape_complexity"
]]

# Save as CSV for training
final_features.to_csv("processed_data/land_use_features.csv", index=False)
print("✅ Feature extraction complete! Saved as land_use_features.csv")
