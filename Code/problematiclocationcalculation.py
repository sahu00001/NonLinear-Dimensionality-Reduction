###############Problematicnode_calculation##################################
import h5py
import numpy as np
import pandas as pd

# Load the data from the .mat file using h5py
with h5py.File('NACCS_testDATAPCAwith72compo30000locations.mat', 'r') as test:
    Resp_test = test['Resp_test'][:]
    Y_hat = test['Y_hat'][:]
    param = test['Param_test'][:]

# Debug: Print shapes of loaded arrays
print("Shapes before transposing:")
print("Resp_test shape:", Resp_test.shape)
print("Y_hat shape:", Y_hat.shape)
print("param shape:", param.shape)

# Ensure the dimensions are correctly aligned if needed (transposing might be necessary)
Resp_test = np.transpose(Resp_test, axes=(2, 1, 0))
Y_hat = np.transpose(Y_hat, axes=(2, 1, 0))

# Debug: Print shapes after transposing
print("Shapes after transposing:")
print("Resp_test shape:", Resp_test.shape)
print("Y_hat shape:", Y_hat.shape)

# Define the number of top differences to find
top_n = 9
num_storms = Resp_test.shape[0]

# Initialize an array to store the top locations for each stormproblematiclocationcalculation
top_locations_array = np.zeros((top_n, num_storms), dtype=int)

# Iterate through each storm and calculate differences
for storm in range(num_storms):
    df = np.zeros(Resp_test.shape[2])  # Assuming third dimension size
    
    # Calculate the sum of absolute differences for each location
    for i in range(Resp_test.shape[2]):
        diff = np.sum(np.abs(np.squeeze(Resp_test[storm, :, i]) - np.squeeze(Y_hat[storm, :, i])))
        df[i] = diff
        # Debug: Print intermediate results
        if i % 1000 == 0:  # Print every 1000 iterations to avoid too much output
            print(f"Processed storm {storm}, location {i}, difference: {diff}")

    # Find the indices of the top 100 largest differences
    ind = np.argpartition(df, -top_n)[-top_n:]  # Get top 'top_n' indices
    ind = ind[np.argsort(df[ind])[::-1]]  # Sort indices by the largest differences

    # Store the top locations for the current storm
    top_locations_array[:, storm] = ind

    # Debug: Print the indices and their corresponding differences
    print(f"Top differences for storm {storm}:")
    for i in range(top_n):
        print(f"Index: {ind[i]}, Difference: {df[ind[i]]}")

# Return the top locations array
print("Top locations array for all storms:")
print(top_locations_array)
print(top_locations_array.shape)

# Count Occurrences and Track Storms
location_counts = np.bincount(top_locations_array.flatten())
location_storms = {}

for storm in range(top_locations_array.shape[1]):
    for location in top_locations_array[:, storm]:
        if location not in location_storms:
            location_storms[location] = set()
        location_storms[location].add(storm)

# Create and Sort DataFrame
location_data = {
    'Location': [],
    'Count': [],
    'Storms': []
}

for location, count in enumerate(location_counts):
    if count > 0:
        location_data['Location'].append(location)
        location_data['Count'].append(count)
        location_data['Storms'].append(sorted(location_storms[location]))

location_counts_df = pd.DataFrame(location_data)

# Sort the DataFrame by the count in descending order
location_counts_df = location_counts_df.sort_values(by='Count', ascending=False)

# Print and Save Results
print(location_counts_df)

# If you want to save the DataFrame to a CSV file, uncomment the following line:
location_counts_df.to_csv('problametic_nodes.csv', index=False)


########problematic nodes time series plus gp predictions_plot(original/Gp+PCA)#######################
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import h5py
import pandas as pd


# Load the data from the .mat file using h5py
with h5py.File('NACCS_testDATAPCAwith72compo30000locations.mat', 'r') as test:
    Resp_test = test['Resp_test'][:]
    Y_hat = test['Y_hat'][:]
    param = test['Param_test'][:]


# Debug: Print shapes of loaded arrays
print("Shapes before transposing:")
print("Resp_test shape:", Resp_test.shape)
print("Y_hat shape:", Y_hat.shape)
print("param shape:", param.shape)

# Ensure the dimensions are correctly aligned if needed (transposing might be necessary)
Resp_test = np.transpose(Resp_test, axes=(2, 1, 0))
Y_hat = np.transpose(Y_hat, axes=(2, 1, 0))

# Debug: Print shapes after transposing
print("Shapes after transposing:")
print("Resp_test shape:", Resp_test.shape)
print("Y_hat shape:", Y_hat.shape)



# Load disp_vect.mat and extract the displacement vector
disp_vect_data = loadmat('disp_vect.mat')  # Adjust the key name based on the file's structure
disp_vect = disp_vect_data['cand_dis'].flatten()  # Assuming 'cand_dis' contains the vector
print(f"Displacement vector loaded with shape: {disp_vect.shape}")  # (170,)

def plot_across_time_steps(storm_index, location_index):
    """
    Plots Resp_test and Y_hat across all time steps for a given storm and location index.

    Parameters:
    - storm_index: Index of the storm (0-based).
    - location_index: Index of the location (0-based).
    """
    # Ensure Resp_test and Y_hat are defined
    global Resp_test, Y_hat

    # Ensure indices are within bounds
    if storm_index < 0 or storm_index >= Resp_test.shape[0]:
        print(f"Invalid storm_index: {storm_index}. Must be between 0 and {Resp_test.shape[0] - 1}.")
        return
    if location_index < 0 or location_index >= Resp_test.shape[2]:
        print(f"Invalid location_index: {location_index}. Must be between 0 and {Resp_test.shape[2] - 1}.")
        return
    
    # Extract data for the given storm and location
    resp_values = Resp_test[storm_index, :, location_index]
    y_hat_values = Y_hat[storm_index, :, location_index]

    # Plot data
    plt.figure(figsize=(10, 6))
    plt.plot(resp_values, label="Resp_test", linestyle='--')
    plt.plot(y_hat_values, label="Y_hat", linestyle='-')
    plt.xlabel("t(hr)")
    plt.ylabel("surge(m)")
    plt.title(f"Storm Index: {storm_index}, Location Index: {location_index}")
    plt.legend()
    plt.grid(False)
    
    svg_filename = f"plot_storm{storm_index}_location{location_index}.svg"
    plt.savefig(svg_filename, format='svg')
    print(f"Plot saved as: {svg_filename}")
    
    plt.show()
    
    




    
    


##############################latlongextractions##############################
# import pandas as pd
# import matplotlib.pyplot as plt
# import contextily as ctx  # To add a basemap

# # Load the two CSV files
# location_counts_df = pd.read_csv('location_countsnew.csv')  # Ensure this file exists in the directory
# centroids_df = pd.read_csv('centroids_and_closest_nodes_updated.csv')  # Ensure this file exists in the directory

# # Extract the location indices from the location_countsnew.csv file
# location_indices = location_counts_df['Location'].values

# # Filter rows in centroids_and_closest_nodes_updated.csv based on Cluster_Number
# filtered_centroids = centroids_df[centroids_df['Cluster_Number'].isin(location_indices)]

# # Retrieve the node_latitude and node_longitude columns
# latitudes = filtered_centroids['Node_Latitude']
# longitudes = filtered_centroids['Node_Longitude']

# # Create the map plot
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.scatter(longitudes, latitudes, c='blue', alpha=0.7)

# # Add basemap using contextily without attribution
# ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, attribution=False)

# # Customize the map (optional)
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')

# # Show the plot
# plt.show()
