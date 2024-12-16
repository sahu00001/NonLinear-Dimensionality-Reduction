import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the storm surge data
with h5py.File('selected535STORMS_NACCS.mat', 'r') as file:
    resp_final_dis = np.array(file['Resp_train'])  # Shape: (170, 12603, 535)
print("Original storm surge data shape:", resp_final_dis.shape)

# plt.plot(resp_final_dis[0, :, 0])
# plt.title("Storm Surge at Location 1 for Storm 1")
# plt.xlabel("Time Stamps")
# plt.ylabel("Storm Surge Value")
# plt.grid(True)
# plt.show()

num_locations = resp_final_dis.shape[0]
num_time_steps = resp_final_dis.shape[1]
num_storms = resp_final_dis.shape[2]
flattened_manual = np.zeros((num_locations, num_time_steps * num_storms))
for loc in range(num_locations):
    flattened_row = []
    for storm in range(num_storms):
        # Collect all time steps for each storm and append them
        flattened_row.extend(resp_final_dis[loc, :, storm])
    # Convert the list to a numpy array and store it
    flattened_manual[loc, :] = np.array(flattened_row)

print("Manual flattened data shape:", flattened_manual.shape)


print("Flattened data shape:", flattened_manual.shape)

# Load the grid data (latitude and longitude) and transpose it as requested
with h5py.File('NACCS.mat', 'r') as file:
    grid_table = np.array(file['grid'])  
    grid_table_transposed = np.transpose(grid_table)
print(f"Shape of grid_table after transposition: {grid_table_transposed.shape}")  

latitudes = grid_table_transposed[:, 0]  # First column is latitude
longitudes = grid_table_transposed[:, 1]  # Second column is longitude
print(f"Latitude shape: {latitudes.shape}, Longitude shape: {longitudes.shape}")  # Expected shape: (12603,)

# Perform PCA
pca = PCA()
eigenvectors = pca.fit_transform(flattened_manual)  
print('The shape of eigenvectors is:', eigenvectors.shape)  # Should be (12603, n_components)
eigenvectors = eigenvectors[:, :10]  # First 10 components for dimensionality reduction

eigenvalues = pca.explained_variance_[:10]  
print(f"First 10 eigenvalues: {eigenvalues}")

# Define a_n (scaling factor)
a_n = 1  # Replace with actual value if necessary

# Calculate 'a' using the formula
a = (np.sum(eigenvalues) / np.sqrt(2)) * a_n
print(f"Calculated 'a': {a}")

# Calculate individual weights for latitude and longitude at each location
mu_lat = np.mean(latitudes)
mu_long = np.mean(longitudes)

w_lat = a / ((latitudes - mu_lat) ** 2)  # Weight for latitude at each location
w_long = a / ((longitudes - mu_long) ** 2)  # Weight for longitude at each location

print(f"w_lat shape: {w_lat.shape}, w_long shape: {w_long.shape}")

# Calculate weights for eigenvectors
w_eigenvectors = np.zeros_like(eigenvectors)
for g in range(10):
    mu_eigenvector = np.mean(eigenvectors[:, g])
    var_eigenvector = np.var(eigenvectors[:, g])  # Variance of each eigenvector across locations
    w_eigenvectors[:, g] = eigenvalues[g] / ((eigenvectors[:, g] - mu_eigenvector) ** 2)

print(f"w_eigenvectors shape: {w_eigenvectors.shape}")  # Should be (12603, 10)

# Combine the data into a DataFrame for clustering
weighted_features_df = pd.DataFrame({
    'Latitude': latitudes,
    'Longitude': longitudes,
    'Weighted_Latitude': w_lat,
    'Weighted_Longitude': w_long
})

# Add the first 10 weighted eigenvectors to the DataFrame
for g in range(10):
    weighted_features_df[f'Eigenvector_{g+1}'] = eigenvectors[:, g]
    weighted_features_df[f'Weighted_Eigenvector_{g+1}'] = w_eigenvectors[:, g]

print("Shape of weighted_features_df:", weighted_features_df.shape)

# Apply K-Means Clustering on the weighted features
n_clusters = 3000  # Choose the number of clusters
kmeans = KMeans(n_clusters=n_clusters)

# Fit the K-Means model to the weighted features
weighted_features_df['Cluster'] = kmeans.fit_predict(weighted_features_df)

# Get the centroids of the clusters
centroids = kmeans.cluster_centers_
print("Centroids shape:", centroids.shape)

# Extract latitude and longitude from the centroids (first two columns correspond to lat/long in weighted_features_df)
centroid_latitudes = centroids[:, 0]
centroid_longitudes = centroids[:, 1]

# Create a DataFrame with the centroid coordinates and their cluster numbers
centroid_df = pd.DataFrame({
    'Cluster_Number': np.arange(n_clusters),
    'Centroid_Latitude': centroid_latitudes,
    'Centroid_Longitude': centroid_longitudes
})

# Initialize a DataFrame to store the closest nodes to each centroid
closest_nodes = pd.DataFrame(columns=['Cluster_Number', 'Node_Index', 'Node_Latitude', 'Node_Longitude', 'Distance'])

# Calculate the nearest node to each centroid for each cluster
for cluster in range(n_clusters):
    # Filter the nodes in the current cluster
    cluster_nodes = weighted_features_df[weighted_features_df['Cluster'] == cluster]
    
    # Get the centroid coordinates for the current cluster
    centroid_lat = centroid_df.loc[centroid_df['Cluster_Number'] == cluster, 'Centroid_Latitude'].values[0]
    centroid_long = centroid_df.loc[centroid_df['Cluster_Number'] == cluster, 'Centroid_Longitude'].values[0]
    
    # Calculate Euclidean distance between each node in the cluster and the centroid
    distances = np.sqrt((cluster_nodes['Latitude'] - centroid_lat) ** 2 + (cluster_nodes['Longitude'] - centroid_long) ** 2)
    
    # Get the index of the closest node
    min_distance_idx = distances.idxmin()
    
    new_row = pd.DataFrame([{
    'Cluster_Number': cluster,
    'Node_Index': min_distance_idx,
    'Node_Latitude': cluster_nodes.loc[min_distance_idx, 'Latitude'],
    'Node_Longitude': cluster_nodes.loc[min_distance_idx, 'Longitude'],
    'Distance': distances.min()
    }])

    closest_nodes = pd.concat([closest_nodes, new_row], ignore_index=True)

# Save the centroid coordinates and closest nodes information to a CSV file
closest_nodes.to_csv('centroids_and_closest_nodes2.csv', index=False)
print("Centroids and closest nodes saved to 'centroids_and_closest_nodes.csv'")

# Save the clustered DataFrame to a CSV file
weighted_features_df.to_csv('clustered_weighted_data2.csv', index=False)
print("Clustered data saved to 'clustered_weighted_data.csv'")

# Save the eigenvectors with cluster information if needed
np.save('weighted_eigenvectors_with_clusters.npy', eigenvectors)


####################run the above algorithm twice to get the to set of data file of centroids_and_closest_nodes2.csv###########
#####below comparing the sheets