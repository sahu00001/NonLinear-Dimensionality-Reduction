from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import h5py
import pandas as pd

# Load and preprocess the data
with h5py.File('clusteredTEST_datasetNACCS.mat', 'r') as f_input:
    resp_final_dist = np.array(f_input['Resp_test_clust'])  # Shape: (3000, 170, 535)

print(f"Original shape: {resp_final_dist.shape}")  # (3000, 170, 535)

resp_final_dist_transposed = np.transpose(resp_final_dist, (2, 1, 0))
print(f"Transposed: {resp_final_dist_transposed.shape}")  # (535, 170, 3000)

# Load disp_vect.mat and extract the displacement vector
disp_vect_data = loadmat('disp_vect.mat')  # Adjust the key name based on the file's structure
disp_vect = disp_vect_data['cand_dis'].flatten()  # Assuming 'cand_dis' contains the vector
print(f"Displacement vector loaded with shape: {disp_vect.shape}")  # (170,)


# Function to plot time series for a given storm and node index
def plot_time_series(storm_idx, node_idx):
    """
    Plots the time series data for a specific storm (location) and node.

    Parameters:
    - storm_idx: Index of the storm (0 to 534).
    - node_idx: Index of the node (0 to 2999).
    """
    # Validate inputs
    n_storms, n_timesteps, n_nodes = resp_final_dist_transposed.shape
    if storm_idx >= n_storms or node_idx >= n_nodes:
        print(f"Invalid indices: Storm index {storm_idx} or Node index {node_idx}.")
        return
    if len(disp_vect) != n_timesteps:
        print("Error: Displacement vector length does not match the number of time steps.")
        return

    # Extract time series data
    time_series = resp_final_dist_transposed[storm_idx, :, node_idx]

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(disp_vect, time_series, label=f"Storm {storm_idx}, Node {node_idx}", linestyle='-', color='blue')
    plt.title(f"Time Series for Storm {storm_idx} at Node {node_idx}")
    plt.xlabel("Displacement(km) ")
    plt.ylabel("Surge(m)")
    plt.legend()
    plt.grid(True)

    # Save and display the plot
    plot_filename = f"time_series_storm{storm_idx}_node{node_idx}.svg"
    plt.savefig(plot_filename, format='svg')
    plt.show()
    print(f"Plot saved as {plot_filename}")


# Example usage
#plot_time_series(56, 1000)  # Replace with desired storm and node indices
