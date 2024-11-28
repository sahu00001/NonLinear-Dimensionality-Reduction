# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# import contextily as ctx
# from matplotlib.ticker import FuncFormatter
# from matplotlib.colors import LinearSegmentedColormap

# # Open the .mat file
# with h5py.File('NACCS_databaseWETprocessed.mat', 'r') as f_input:
#     # Access datasets
#     resp_final_dis = np.array(f_input['RespFinaldis'])
#     grid = np.array(f_input['grid'])

# # Extract latitude and longitude from grid
# latitudes = grid[0, :]
# longitudes = grid[1, :]

# # Select data for 1 storm and 1 timestamp
# storm_index = 32
# time_index = 169
# data = resp_final_dis[:, time_index, storm_index]

# # Define the indices of interest
# highlight_indices = [9482, 10653, 11553, 8519]

# # Define a formatter function for tick labels with degree symbols
# def degree_formatter(x, _):
#     return f"{x:.0f}°"

# # Use the 'jet' colormap
# custom_cmap = plt.cm.get_cmap('jet')

# # Plot with ESRI Basemap
# fig, ax = plt.subplots(figsize=(12, 10))

# # Scatter plot of data using the custom colormap
# sc = ax.scatter(longitudes, latitudes, c=data, cmap=custom_cmap, marker='o', s=10)

# # Add a smaller colorbar
# cbar = plt.colorbar(sc, ax=ax, label="Surge (m)", orientation='vertical', shrink=0.75, aspect=30)

# # Add ESRI basemap without attribution
# ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, attribution='')

# # Add grid lines
# ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.9)

# # Set axis limits
# ax.set_xlim([-78, -67])
# ax.set_ylim([36, latitudes.max()])

# # Apply the degree formatter to both axes
# ax.xaxis.set_major_formatter(FuncFormatter(degree_formatter))
# ax.yaxis.set_major_formatter(FuncFormatter(degree_formatter))

# # Add axis labels
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")

# # Highlight specific points with smaller square boxes
# for idx in highlight_indices:
#     ax.scatter(longitudes[idx], latitudes[idx], color='white', s=40, marker='s', edgecolors='white', linewidths=0.8)
#     ax.annotate(f"SP #{idx}", (longitudes[idx], latitudes[idx]),
#                 textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8, fontweight="bold", color='white')
    
# plt.savefig("databasescatterplot.svg", format="svg", bbox_inches='tight')

# # Show the plot
# plt.show()













# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# import contextily as ctx
# from matplotlib.ticker import FuncFormatter

# # Open the .mat file
# with h5py.File('NACCS_databaseWETprocessed.mat', 'r') as f_input:
#     # Access datasets
#     resp_final_dis = np.array(f_input['RespFinaldis'])  # Shape: (12603, 170, 595)
#     grid = np.array(f_input['grid'])  # Shape: (3, 12603)

# # Extract latitude and longitude from grid
# latitudes = grid[0, :]  # First row is latitude
# longitudes = grid[1, :]  # Second row is longitude

# # Select data for 1 storm (e.g., storm index 23) and 1 timestamp (e.g., time index 100)
# storm_index = 23
# time_index = 100
# data = resp_final_dis[:, time_index, storm_index]  # Shape: (12603,)

# # Define a formatter function for tick labels with degree symbols
# def degree_formatter(x, _):
#     return f"{x:.0f}°"

# # Plot with ESRI Basemap
# fig, ax = plt.subplots(figsize=(12, 10))

# # Scatter plot of data
# sc = ax.scatter(longitudes, latitudes, c=data, cmap='coolwarm', marker='o', s=10)

# # Add a smaller colorbar
# cbar = plt.colorbar(sc, ax=ax, label="Surge(m)", orientation='vertical', shrink=0.75, aspect=30)

# # Add ESRI basemap without attribution
# ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, attribution='')

# # Add grid lines
# ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.9)

# # Set axis limits
# ax.set_xlim([-78, -67])  # Cut x-axis to range from -80 to -65
# ax.set_ylim([36, latitudes.max()])  # Start y-axis at 36 degrees

# # Apply the degree formatter to both axes
# ax.xaxis.set_major_formatter(FuncFormatter(degree_formatter))
# ax.yaxis.set_major_formatter(FuncFormatter(degree_formatter))

# # Add axis labels
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")

# # Show the plot
# plt.show()

# import h5py
# import numpy as np
# import matplotlib.pyplot as plt

# # Open the .mat file
# with h5py.File('NACCS_databaseWETprocessed.mat', 'r') as f_input:
#     # Access datasets
#     resp_final_dis = np.array(f_input['RespFinaldis'])  # Shape: (12603, 170, 595)

# # Specify the storm index and SPs to analyze
# storm_index = 30  # 24th storm in the dataset
# sp_indices = [9482, 10653, 11553, 8519]  # Specific SPs as shown in the example

# # Extract time series data for the storm at specified SPs, limited to 102 time steps
# time_indices = np.arange(170)  # Restrict to first 102 time steps
# time_series_data = resp_final_dis[sp_indices, :170, storm_index]  # Shape: (len(sp_indices), 102)

# # Define colors for the SPs
# colors = ['blue', 'red', 'orange', 'green']
# labels = [f"SP #{sp}" for sp in sp_indices]

# # Plot the time series for each SP
# plt.figure(figsize=(10, 7))
# for i, sp in enumerate(sp_indices):
#     plt.plot(time_indices, time_series_data[i, :], label=labels[i], color=colors[i], linewidth=1.5)

# # Add labels, legend, and title
# plt.title("(b) Time-series for a specific storm simulation", fontsize=14)
# plt.xlabel("Time Step")
# plt.ylabel("Response Value")
# plt.legend(loc="upper left", fontsize=10)
# plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# # Show the plot
# plt.show()


# import h5py
# import numpy as np
# import matplotlib.pyplot as plt

# # Open the .mat file
# with h5py.File('NACCS_databaseWETprocessed.mat', 'r') as f_input:
#     # Access datasets
#     resp_final_dis = np.array(f_input['RespFinaldis'])  # Shape: (12603, 170, 595)
#     cand_dis = np.array(f_input['cand_dis'])  # Shape: (170,) or similar

# # Specify the storm index and SPs to analyze
# storm_index = 32 
# sp_indices = [9482, 10653, 11553, 8519]  # Specific SPs as shown in the example

# # Extract time series data for the storm at specified SPs
# time_series_data = resp_final_dis[sp_indices, :, storm_index]  # Shape: (len(sp_indices), 170)

# # Use `cand_dis` for the x-axis
# x_values = cand_dis  # Assumes cand_dis corresponds to 170 time steps

# # Define colors for the SPs
# colors = ['blue', 'red', 'orange', 'green']
# labels = [f"SP #{sp}" for sp in sp_indices]

# # Plot the time series for each SP
# plt.figure(figsize=(10, 7))
# for i, sp in enumerate(sp_indices):
#     plt.plot(x_values, time_series_data[i, :], label=labels[i], color=colors[i], linewidth=1.5)

# # Add labels, legend, and title
# plt.title("(b) Time-series for a specific storm simulation", fontsize=14)
# plt.xlabel("Distance to Landfall(km)")
# plt.ylabel("Surge(m)")
# plt.legend(loc="upper left", fontsize=10)
# plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# plt.savefig("databasetimeseriesplot.svg", format="svg")

# # Show the plot
# plt.show()
################################dr pi####################
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# # Load the .mat file and access the dataset
# with h5py.File('NACCS_databaseWETprocessed.mat', 'r') as f_input:
#     # Access datasets
#     resp_final_dis = np.array(f_input['RespFinaldis'])

# # Print the shape of the dataset
# print("Shape of RespFinaldis:", resp_final_dis.shape)

# location_index = 1  # Change this to the desired location index (0-based)
# timestamp_index = 168  # Change this to the desired timestamp index (0-based)

# # Extract the data for the specified location and timestamp across all storms
# data_to_plot = resp_final_dis[location_index, timestamp_index, :]

# # Sort the data in increasing order along with the storm indices
# sorted_indices = np.argsort(data_to_plot)  # Get indices that would sort the array
# sorted_data = data_to_plot[sorted_indices]  # Sort data using the indices

# # Plot the sorted data
# plt.figure(figsize=(10, 6))
# plt.plot(range(len(sorted_data)), sorted_data, marker='o')
# plt.title(f"Location {location_index + 1}, Timestamp {timestamp_index + 1}")
# plt.xlabel("Storms")
# plt.ylabel("Surge(m)")
# plt.grid(False)
# plt.show()

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
location_index = 400  # Change this to the desired location index (0-based)
timestamp_index = 90  # Change this to the desired timestamp index (0-based)

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
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

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

