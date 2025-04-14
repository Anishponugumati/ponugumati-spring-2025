import os
import glob
import geopandas as gpd
import pandas as pd

# Directory paths (adjust these paths to your folder structure)
input_dir = "geojson_with_elevation2"  # Directory containing the enriched GeoJSON files
output_csv = "Land_use_data.csv"

# Mapping dictionary: NLCD land cover (code) --> Inferred zoning label
# These mappings are a heuristic based on common interpretations.
# You can refine these labels as needed:
nlcd_to_zoning = {
    21: "Residential",          # Developed, Open Space (often low-density or parks)
    22: "Residential",          # Developed, Low Intensity (typically residential)
    23: "Commercial",           # Developed, Medium Intensity (often commercial/institutional)
    24: "Commercial",           # Developed, High Intensity (mostly commercial/mixed-use)
    41: "Recreational/Green",    # Deciduous Forest (used as parks, preserves)
    42: "Recreational/Green",    # Evergreen Forest
    43: "Recreational/Green",    # Mixed Forest
    52: "Agricultural",         # Shrub/Scrub
    71: "Agricultural",         # Grassland/Herbaceous (can be agricultural in rural fringe)
    81: "Agricultural",         # Pasture/Hay
    82: "Agricultural",         # Cultivated Crops
    90: "Wetlands",             # Woody Wetlands
    95: "Wetlands"              # Emergent Herbaceous Wetlands
}

# Function to infer zoning label from the NLCD land_cover code
def infer_zoning(land_cover):
    return nlcd_to_zoning.get(land_cover, "Unknown")

# List to hold DataFrames for each city
dfs = []

# Build a list of all enriched GeoJSON files (assumed naming convention: CityName_USA_NLCD_with_elevation.geojson)
geojson_files = glob.glob(os.path.join(input_dir, "*_USA_NLCD_with_elevation.geojson"))

for file in geojson_files:
    # Load the GeoJSON as a GeoDataFrame
    gdf = gpd.read_file(file)
    
    # Convert the properties (i.e., all fields except geometry) into a DataFrame
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    
    # Apply the zoning mapping
    df["zoning_label"] = df["land_cover"].apply(infer_zoning)
    
    # Extract city name from the file name (assumes file name structure: CityName_USA_NLCD_with_elevation.geojson)
    city = os.path.basename(file).split("_USA_NLCD_with_elevation")[0]
    df["city"] = city  # Reinforce the city column if needed
    
    # Append to our list
    dfs.append(df)

# Concatenate the DataFrames from all cities into one master DataFrame
# Before concatenating, drop columns that are entirely NA from each DataFrame
cleaned_dfs = [df.dropna(axis=1, how='all') for df in dfs]

consolidated_df = pd.concat(cleaned_dfs, ignore_index=True)


# Save to CSV for model training or further analysis
consolidated_df.to_csv(output_csv, index=False)
print(f"Consolidated data with zoning labels saved to {output_csv}")
