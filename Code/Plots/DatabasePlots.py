

#################################timeseries plot ###########################

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Open the .mat file
with h5py.File('NACCS_databaseWETprocessed.mat', 'r') as f_input:
    # Access datasets
    resp_final_dis = np.array(f_input['RespFinaldis'])  # Shape: (12603, 170, 595)
    cand_dis = np.array(f_input['cand_dis'])  # Shape: (170,) or similar

# Specify the storm index and SPs to analyze
storm_index = 32 
sp_indices = [9482, 10653, 11553, 8519]  # Specific SPs as shown in the example

# Extract time series data for the storm at specified SPs
time_series_data = resp_final_dis[sp_indices, :, storm_index]  # Shape: (len(sp_indices), 170)

# Use `cand_dis` for the x-axis
x_values = cand_dis  # Assumes cand_dis corresponds to 170 time steps

# Define colors for the SPs
colors = ['blue', 'red', 'orange', 'green']
labels = [f"SP #{sp}" for sp in sp_indices]

# Plot the time series for each SP
plt.figure(figsize=(10, 7))
for i, sp in enumerate(sp_indices):
    plt.plot(x_values, time_series_data[i, :], label=labels[i], color=colors[i], linewidth=1.5)

# Set x-axis limits
plt.xlim([-2000, 500])

# Add labels, legend, and title
plt.title("(b) Time-series for a specific storm simulation", fontsize=14)
plt.xlabel("Distance to Landfall(km)")
plt.ylabel("Surge(m)")
plt.legend(loc="upper left", fontsize=10)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Save and show the plot
plt.savefig("databasetimeseriesplot.svg", format="svg")
plt.show()

################################Linearity and non Linearity####################

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load the .mat file and access the dataset
with h5py.File('NACCS_databaseWETprocessed.mat', 'r') as f_input:
    # Access datasets
    resp_final_dis = np.array(f_input['RespFinaldis'])

# Print the shape of the dataset
print("Shape of RespFinaldis:", resp_final_dis.shape)

# Specify location and timestamp
location_index = 34  # Change this to the desired location index (0-based)
timestamp_index = 120  # Change this to the desired timestamp index (0-based)

# Extract the data for the specified location and timestamp across all storms
data_to_plot = resp_final_dis[location_index, timestamp_index, :]

# Sort the data in increasing order along with the storm indices
sorted_indices = np.argsort(data_to_plot)  # Get indices that would sort the array
sorted_data = data_to_plot[sorted_indices]  # Sort data using the indices

# Define the ranges for non-linear and linear regions
non_linear_start = 100  # End of the non-linear region at the start
non_linear_end = len(sorted_data) - 100  # Start of the non-linear region at the end

# Plot the sorted data
plt.figure(figsize=(12, 8))  # Larger figure size for clarity
plt.plot(range(len(sorted_data)), sorted_data, marker='o', markersize=5, 
          linestyle='-', linewidth=1.5, color='royalblue', label="Response Values")

# Add titles and labels with improved formatting
#plt.title(f"Linearity and Non-Linearity Analysis",fontsize=16, fontweight='bold')
plt.xlabel("Storms", fontsize=14)
plt.ylabel("Surge(m)", fontsize=14)

# Add a grid for better readability
#plt.grid(False, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Customize tick sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add vertical lines to separate linear and non-linear regions
plt.axvline(x=non_linear_start, color='black', linestyle='--', linewidth=1, label="Linear-Nonlinear Boundary")
plt.axvline(x=non_linear_end, color='black', linestyle='--', linewidth=1)

# Label the linear and non-linear regions (position adjusted to avoid overlap)
plt.text(non_linear_start / 2, max(sorted_data) + 0.05, "Non-Linear Region", fontsize=12, color='red', ha='center')
plt.text((non_linear_start + non_linear_end) / 2, max(sorted_data) + 0.05, "Linear Region", fontsize=12, color='green', ha='center')
plt.text((non_linear_end + len(sorted_data)) / 2, max(sorted_data) + 0.05, "Non-Linear Region", fontsize=12, color='red', ha='center')

# Add a legend
plt.legend(fontsize=12, loc='upper left')

# Tighten layout to prevent overlap
plt.tight_layout()

plt.savefig("linearity_non_linearity_plot.svg", format="svg")

# Show the plot
plt.show()

#######################Total 12603 Locations#######################################

import h5py
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.colors import ListedColormap, BoundaryNorm

# Open the .mat file
with h5py.File('NACCS_databaseWETprocessed.mat', 'r') as f_input:
    # Access grid data
    grid = np.array(f_input['grid'])

# Extract latitude and longitude from grid
latitudes = grid[0, :]
longitudes = grid[1, :]

# Define a single color (e.g., red) and set up the color bar
single_color = 'red'  # Desired color
cmap = ListedColormap([single_color])
bounds = [0, 1]  # Single boundary
norm = BoundaryNorm(bounds, cmap.N)

# Create the map plot
fig, ax = plt.subplots(figsize=(12, 10))

# Plot the points using the single color
scatter = ax.scatter(longitudes, latitudes, c=[0.5] * len(latitudes), cmap=cmap, norm=norm, s=10, alpha=0.7, marker='o')

# Add ESRI basemap without attribution
ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, attribution='')

# Reapply axis limits after adding the basemap
ax.set_xlim([-78, -67])  # Adjust these limits as needed
ax.set_ylim([36, latitudes.max()])  # Ensure y-axis starts at 36

# Add axis labels
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")

# Add a color bar (even though all points are red)
cbar = plt.colorbar(scatter, ax=ax, boundaries=bounds, ticks=[0.5], orientation='vertical', pad=0.01, aspect=30)
cbar.set_label('Arbitrary Color Bar', fontsize=12)
cbar.ax.set_yticklabels(['Red'], fontsize=10)  # Labeling the single color

# Save the plot as an SVG file
output_filename = "alllocationscolourbar.svg"
plt.savefig(output_filename, format='svg', bbox_inches='tight')

# Show the plot
plt.show()

######################3000 locations#######################################

import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.colors import ListedColormap, BoundaryNorm

# Load the centroids_and_closest_nodes_updated.csv file
centroids_df = pd.read_csv('centroids_and_closest_nodes_updated.csv')  # Ensure this file exists in the directory

# Extract latitude and longitude
latitudes = centroids_df['Node_Latitude']
longitudes = centroids_df['Node_Longitude']

# Define a single color (e.g., red) and set up the color bar
single_color = 'blue'  # Desired color
cmap = ListedColormap([single_color])
bounds = [0, 1]  # Single boundary
norm = BoundaryNorm(bounds, cmap.N)

# Create the map plot
fig, ax = plt.subplots(figsize=(12, 10))

# Plot the points using the single color
scatter = ax.scatter(longitudes, latitudes, c=[0.5] * len(latitudes), cmap=cmap, norm=norm, s=10, alpha=0.7, marker='o', label='Locations')

# Add ESRI basemap without attribution
ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, attribution='')

# Reapply axis limits after adding the basemap
ax.set_xlim([-78, -67])  # Adjust these limits as needed
ax.set_ylim([36, latitudes.max()])  # Ensure y-axis starts at 36

# Add axis labels
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")

# Add a legend
plt.legend(loc='lower left', fontsize=10, title='Legend', title_fontsize=12)

# Add a color bar (even though all points are red)
cbar = plt.colorbar(scatter, ax=ax, boundaries=bounds, ticks=[0.5], orientation='vertical', pad=0.01, aspect=30)
cbar.set_label('Arbitrary Color Bar', fontsize=12)
cbar.ax.set_yticklabels(['Red'], fontsize=10)  # Labeling the single color

# Save the plot as an SVG file
output_filename = "centroids_single_color_map_with_legend.svg"
plt.savefig(output_filename, format='svg', bbox_inches='tight')

# Show the plot
plt.show()

#########################12603 Location divided into 3000 Clusters###########
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
from matplotlib.colors import ListedColormap

# Load the original_points_with_clusters.csv file
points_df = pd.read_csv('original_points_with_clusters.csv')

# Extract latitude, longitude, and cluster information
latitudes = points_df['Latitude']
longitudes = points_df['Longitude']
clusters = points_df['Cluster']

# Generate 3,000 distinct colors using HSV
num_clusters = clusters.nunique()  # Number of unique clusters
np.random.seed(42)  # For reproducibility
unique_colors = plt.cm.hsv(np.linspace(0, 1, num_clusters))  # HSV space evenly distributed
cmap = ListedColormap(unique_colors)

# Plot the map
fig, ax = plt.subplots(figsize=(12, 10))

# Plot the points with unique cluster colors
scatter = ax.scatter(
    longitudes,
    latitudes,
    c=clusters,
    cmap=cmap,
    s=10,  # Dot size
    alpha=0.7,
    marker='o'
)

# Add ESRI basemap without attribution
ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, attribution='')

# Set axis limits
ax.set_xlim([-78, -67])
ax.set_ylim([36, latitudes.max()])

# Add axis labels
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")

# Add a color bar for clusters
cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.01, aspect=30)
cbar.set_label('Cluster', fontsize=12)
cbar.ax.tick_params(labelsize=10)

# Save the plot
output_filename = "original_points_3000_distinct_colors_map.svg"
plt.savefig(output_filename, format='svg', bbox_inches='tight')

# Show the plot
plt.show()
