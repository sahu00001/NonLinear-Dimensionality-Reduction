import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point

# Load the dataset
data = pd.read_csv('matched_results.csv')

# Extract Latitude and Longitude columns
latitude = data['Latitude']
longitude = data['Longitude']

# Create a GeoDataFrame
geometry = [Point(xy) for xy in zip(longitude, latitude)]
gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")  # Use WGS 84 (EPSG:4326)

# Plot the data
fig, ax = plt.subplots(figsize=(12, 8))
gdf.plot(ax=ax, color='red', markersize=5, label='Locations')

# Add basemap using contextily
ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, attribution='')

# Customize the plot
ax.set_title("Problematic locations", fontsize=16)
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()
