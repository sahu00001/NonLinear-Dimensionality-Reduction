import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import h5py
import pandas as pd

# Load and preprocess the data
with h5py.File('clusteredTEST_datasetNACCS.mat', 'r') as f_input:
    resp_final_dist = np.array(f_input['Resp_test_clust'])  # Shape: (3000, 170, 535)

print(f"Original shape: {resp_final_dist.shape}")  # (3000, 170, 535)

# Transpose data to match the desired shape for Resp3_restored
resp_final_dist_transposed = np.transpose(resp_final_dist, (2, 1, 0))  # Shape: (535, 170, 3000)

# Define parameters
n_storms = 535
n_nodes = 3000
n_steps = 170

# Flatten the data for PCA or scaling
flattened_data = np.zeros((60, 170 * 3000))
for storm_idx in range(60):
    flattened = []
    for loc_idx in range(3000):
        flattened.extend(resp_final_dist_transposed[storm_idx, :, loc_idx])
    flattened_data[storm_idx, :] = flattened

# Apply PCA and reconstruct data from principal components
pca_full = PCA(n_components = 216)
pca_full_data = pca_full.fit_transform(flattened_data)
data_reconstructed = pca_full.inverse_transform(pca_full_data)

# Standardize the data
scaler = StandardScaler()
scaler.fit(flattened_data)

# Load reconstructed data from CSVs
data_auto1 = pd.read_csv('reconstructed_56std.csv', header=None)
data_auto2 = pd.read_csv('reconstructed_56nonstd.csv', header=None)
data_de_standardized = (data_auto1 * scaler.scale_) + scaler.mean_

# Convert to numpy arrays
Resp1 = data_de_standardized.to_numpy()
#Resp2 = data_auto2.to_numpy()

# Initialize restored arrays
Resp1_restored = np.zeros((n_storms, n_steps, n_nodes))
#Resp2_restored = np.zeros((n_storms, n_steps, n_nodes))
Resp3_restored = resp_final_dist_transposed[:n_storms, :, :n_nodes]  # Use transposed original data
Resp4_restored = np.zeros((n_storms, n_steps, n_nodes))  # PCA-reconstructed data

# Reverse the flattening process for Resp1, Resp2, and Resp4
for storm_idx in range(n_storms):
    for loc_idx in range(n_nodes):
        start_idx = loc_idx * n_steps
        end_idx = start_idx + n_steps
        Resp1_restored[storm_idx, :, loc_idx] = Resp1[storm_idx, start_idx:end_idx]
        #Resp2_restored[storm_idx, :, loc_idx] = Resp2[storm_idx, start_idx:end_idx]
        Resp4_restored[storm_idx, :, loc_idx] = data_reconstructed[storm_idx, start_idx:end_idx]

# Define combinations of storm and node indices to plot
combinations = [
    (56, 1000), (44, 2500), (45, 300), (23, 400), (52, 1500),
    (27, 2000), (14, 700), (36, 1000), (54, 900), (13, 1000),
    (36, 1100), (36, 1200), (44, 1300), (14, 1400), (37, 1500),
    (14, 1600), (15, 1700), (14, 1800), (10, 1900), (25, 2000)
]

# Loop through each specific combination and plot
for storm_idx, node_idx in combinations:
    # Extract data for the specified storm and node
    resp1_data = Resp1_restored[storm_idx, :, node_idx]
    #resp2_data = Resp2_restored[storm_idx, :, node_idx]
    resp3_data = Resp3_restored[storm_idx, :, node_idx]
    resp4_data = Resp4_restored[storm_idx, :, node_idx]  # PCA-reconstructed data
    
    # Plot to compare all datasets for the specified storm and node
    plt.figure(figsize=(12, 6))
    plt.plot(resp1_data, label="Destandardized Data", linestyle='-')
    #plt.plot(resp2_data, label="Non-standardized Data", linestyle='-')
    plt.plot(resp3_data, label="Original Data", linestyle='-')
    plt.plot(resp4_data, label="PCA Reconstructed Data", linestyle='--')
    
    # Labeling
    plt.title(f"Comparison of Restored Data for Storm {storm_idx} at Node {node_idx}")
    plt.xlabel("Timesteps")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(False)

    # Save each plot with a unique filename based on storm and node indices
    plot_filename = f"comparison_plot_storm{storm_idx}_node{node_idx}.svg"
    plt.savefig(plot_filename, format="svg")
    plt.show()
