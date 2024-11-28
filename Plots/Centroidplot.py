import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.colors import ListedColormap

# Load the data
file_path = 'centroids_and_closest_nodes_updated.csv'
data = pd.read_csv(file_path)

# Convert latitude and longitude columns to GeoDataFrame geometry
gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['Node_Longitude'], data['Node_Latitude']))

# Define the coordinate reference system (CRS) as WGS84
gdf.set_crs(epsg=4326, inplace=True)

# Convert CRS to Web Mercator for compatibility with basemaps
gdf = gdf.to_crs(epsg=3857)

# Generate a large list of unique colors for 3000 clusters
n_clusters = gdf['Cluster_Number'].nunique()
cmap = plt.cm.get_cmap('tab20', n_clusters)

# Plot the centroids with Earth imagery as the background
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the clusters
gdf.plot(
    ax=ax,
    column='Cluster_Number',  # Use cluster numbers to differentiate
    cmap=cmap,  # Use a categorical colormap
    markersize=10,
    alpha=0.7,
    legend=False,  # Simplify legend
)

# Add ESRI World Imagery basemap
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, attribution=False)

# Adjust the x-axis and y-axis limits (crop from left, right, and bottom)
x_min, x_max = gdf.geometry.x.min(), gdf.geometry.x.max()
y_min, y_max = gdf.geometry.y.min(), gdf.geometry.y.max()
ax.set_xlim(x_min + (x_max - x_min) * 0.1, x_max - (x_max - x_min) * 0.1)  # Crop from left and right
ax.set_ylim(y_min + (y_max - y_min) * 0.1, y_max)  # Crop from the bottom

# Convert axis ticks back to WGS84 (latitude/longitude degrees)
x_ticks = ax.get_xticks()
y_ticks = ax.get_yticks()

# Convert tick locations from Web Mercator to WGS84
x_tick_labels = [f"{coord:.2f}°" for coord in gpd.GeoSeries.from_xy(x_ticks, [y_min]*len(x_ticks), crs=3857).to_crs(4326).x]
y_tick_labels = [f"{coord:.2f}°" for coord in gpd.GeoSeries.from_xy([x_min]*len(y_ticks), y_ticks, crs=3857).to_crs(4326).y]

# Set the tick labels
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_tick_labels)

# Add labels, grid, and title
plt.xlabel('Longitude (°)')
plt.ylabel('Latitude (°)')
plt.title('3000 Representative Locations by Cluster')
plt.grid(False)

# Add a single legend for "Clusters"
handles = [plt.Line2D([0], [0], marker='o', color='gray', label='Clusters', linestyle="", alpha=0.7)]
ax.legend(handles=handles, loc='upper left', title="Legend")

# Save the plot as an SVG file
output_file = '3000centroidlocationsfinal_clusters.svg'
plt.savefig(output_file, format='svg', bbox_inches='tight')

print(f"Plot saved as {output_file}")

# Show the plot
plt.show()

# import h5py
# import matplotlib.pyplot as plt
# import contextily as ctx
# import geopandas as gpd
# from shapely.geometry import Point

# # Load the HDF5 file
# file_name = 'NACCS_databaseWETprocessed.mat'
# with h5py.File(file_name, 'r') as f:
#     grid_database = f['grid'][:].T  # Transpose the data

# # Extract latitude and longitude
# latitude = grid_database[:, 0]
# longitude = grid_database[:, 1]

# # Create GeoDataFrame
# geometry = [Point(xy) for xy in zip(longitude, latitude)]
# gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")  # WGS84 (degrees)

# # Plot the data
# fig, ax = plt.subplots(figsize=(10, 10))
# gdf.plot(ax=ax, color='red', markersize=10, alpha=0.7)

# # Add ESRI World Imagery basemap while keeping the axis in degrees
# ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs=gdf.crs, attribution=False)

# # Set x-axis limits
# crop_lon_min = longitude.min()  # Keep the minimum longitude
# crop_lon_max = -66  # Set the maximum longitude to -66

# # Set y-axis limits
# crop_lat_min = 35.5  # Set the minimum latitude to 35
# crop_lat_max = latitude.max()  # Keep the maximum latitude

# # Apply axis limits
# ax.set_xlim(crop_lon_min, crop_lon_max)
# ax.set_ylim(crop_lat_min, crop_lat_max)

# # Add labels and title
# plt.xlabel('Longitude (°)')
# plt.ylabel('Latitude (°)')
# plt.title('12603 locations ')

# # Save the plot as an SVG file
# output_file = '12603_locations.svg'
# plt.savefig(output_file, format='svg', bbox_inches='tight')
# print(f"Plot with adjusted axes saved as {output_file}")

# # Show the plot
# plt.show()
