# import geopandas as gpd
# import pandas as pd
# import matplotlib.pyplot as plt
# from shapely.geometry import Point
# import contextily as ctx
# import matplotlib.cm as cm
# import matplotlib.colors as colors

# # Load the CSV file containing latitude, longitude, and cluster data
# clustered_weighted_data1 = pd.read_csv("clustered_weighted_data1.csv")

# # Create geometry column based on Latitude and Longitude
# geometry = [Point(xy) for xy in zip(clustered_weighted_data1['Longitude'], clustered_weighted_data1['Latitude'])]

# # Create a GeoDataFrame using the latitude and longitude
# gdf = gpd.GeoDataFrame(clustered_weighted_data1, geometry=geometry, crs="EPSG:4326")

# # Convert to Web Mercator projection (required for basemap)
# gdf = gdf.to_crs(epsg=3857)

# # Create a color gradient based on cluster values
# norm = colors.Normalize(vmin=gdf['Cluster'].min(), vmax=gdf['Cluster'].max())  # Normalize the cluster values
# cmap = cm.viridis  # Using a color gradient (viridis)
# gdf['color'] = gdf['Cluster'].apply(lambda x: cmap(norm(x)))  # Apply gradient to cluster values

# # Plot the clusters with gradient color
# fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# # Plot the points using the gradient color
# gdf.plot(ax=ax, color=gdf['color'], markersize=30)

# # Add a satellite basemap using contextily
# ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, attribution='')

# # Add a color bar for the gradient
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('Cluster Gradient')

# # Add title and axis labels
# plt.title('12603 location Clusters ', fontsize=16)
# plt.xlabel('Longitude', fontsize=12)
# plt.ylabel('Latitude', fontsize=12)

# # Clean layout and show the plot
# plt.tight_layout()
# plt.show()




######################################

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np

# Assuming the data is already loaded into a dataframe
# Example column name: 'lat_long' where data is formatted as 'lat, long'

# Replace 'file_path' with the actual path to your CSV file
data = pd.read_csv("shortest_distances_with_indices.csv")

# Clean the 'closest_Lat_Long' column by removing parentheses and spaces
data['Closest_Lat_Long'] = data['Closest_Lat_Long'].str.replace(r'[()]', '', regex=True)

# Split the 'closest_Lat_Long' column into separate 'latitude' and 'longitude' columns
data[['latitude', 'longitude']] = data['Closest_Lat_Long'].str.split(',', expand=True)

# Convert the new latitude and longitude columns to float
data['latitude'] = data['latitude'].astype(float)
data['longitude'] = data['longitude'].astype(float)

# Create a GeoDataFrame to handle spatial data
gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['longitude'], data['latitude']))

# Define the coordinate reference system (CRS) as WGS84 (used by GPS and Google Maps)
gdf.set_crs(epsg=4326, inplace=True)

# Convert the CRS to Web Mercator (needed for contextily basemap)
gdf = gdf.to_crs(epsg=3857)

# Create the scatter plot with red points
fig, ax = plt.subplots(figsize=(10, 6))
gdf.plot(ax=ax, color='red', markersize=10, alpha=0.8)

# Add a satellite basemap without the attribution (by setting attribution=False)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, attribution=False)

# Set labels and title
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clusters Centroids(3000)')
plt.grid(True)

# Show the plot
plt.show()









##########################



# # Read the CSV files
# df1 = pd.read_csv('clustered_weighted_data1.csv')  # Replace with your actual file path
# df2 = pd.read_csv('centroids_and_closest_nodes1.csv')  # Replace with your actual file path

# # Inspect the column names to ensure the correct columns are being referenced
# print("DF1 columns:", df1.columns)
# print("DF2 columns:", df2.columns)

# # Use the actual column names for latitude and longitude after inspection
# latitude_col_df1 = 'Latitude'  # Replace with the actual column name from df1
# longitude_col_df1 = 'Longitude'  # Replace with the actual column name from df1
# latitude_col_df2 = 'Node_Latitude'  # Replace with the actual column name from df2
# longitude_col_df2 = 'Node_Longitude'  # Replace with the actual column name from df2

# # Function to calculate Euclidean distance
# def euclidean_distance(lat1, lon1, lat2, lon2):
#     return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

# # Function to find the shortest Euclidean distance for each point in df2 with all points in df1
# def find_shortest_euclidean_distances(df1, df2):
#     shortest_distances = []
#     closest_points = []
#     closest_indices = []
    
#     for index2, row2 in df2.iterrows():
#         min_distance = float('inf')
#         closest_point = None
#         closest_index = None
        
#         for index1, row1 in df1.iterrows():
#             distance = euclidean_distance(row2[latitude_col_df2], row2[longitude_col_df2], row1[latitude_col_df1], row1[longitude_col_df1])
            
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_point = (row1[latitude_col_df1], row1[longitude_col_df1])
#                 closest_index = index1  # Store the index of the closest point in df1
        
#         shortest_distances.append(min_distance)
#         closest_points.append(closest_point)
#         closest_indices.append(closest_index)  # Store the index number
    
#     df2['Shortest_Distance'] = shortest_distances
#     df2['Closest_Lat_Long'] = closest_points
#     df2['Closest_Index_in_df1'] = closest_indices  # Add index column for the closest point in df1
    
#     return df2

# # Apply the function to find the shortest distance
# df2_with_distances = find_shortest_euclidean_distances(df1, df2)

# # Save the result to a CSV file
# df2_with_distances.to_csv('shortest_distances_with_indices.csv', index=False)

# # Print the resulting dataframe with closest index numbers
# print(df2_with_distances[['Node_Latitude', 'Node_Longitude', 'Shortest_Distance', 'Closest_Lat_Long', 'Closest_Index_in_df1']])
