

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import h5py
import pandas as pd
# Load and preprocess the data

with h5py.File('clusteredTEST_datasetNACCS.mat', 'r') as f_input:
    resp_final_dist = np.array(f_input['Resp_test_clust'])  # Shape: (3000, 170, 535)

print(f"Original shape: {resp_final_dist.shape}")  # (3000, 170, 535)

resp_final_dist_transposed = np.transpose(resp_final_dist, (2, 1, 0))


combinations = [
    (56, 1000), (44, 2500), (45, 300), (23, 400), (52, 1500),
    (27, 2000), (14, 700), (36, 1000), (54, 900), (13, 1000),
    (36, 1100), (36, 1200), (44, 1300), (14, 1400), (37, 1500),
    (14, 1600), (15, 1700), (14, 1800), (10, 1900), (25, 2000)
]

storm_idx = 12
node_idx = 1500
n_storms = 60
n_nodes = 3000
n_steps = 170

#resp1_data=Resp_test_clust[storm_idx, :, node_idx]


plt.figure(figsize=(12, 6))
plt.plot(resp_final_dist[node_idx, :, storm_idx], label="Original_Data", linestyle='-')
plt.plot(resp_final_dist_transposed[storm_idx, :, node_idx], label="Original_Data", linestyle='--')
plt.show()


flattened_data = np.zeros((60, 170 * 3000))
for storm_idx in range(60):
    flattened = []
    for loc_idx in range(3000):
        flattened.extend(resp_final_dist_transposed[storm_idx, :, loc_idx])
    flattened_data[storm_idx, :] = flattened


# Apply PCA for full data
pca_full = PCA(n_components=50)
pca_full_data = pca_full.fit_transform(flattened_data)

# Reconstruct data from principal components
data_reconstructed = pca_full.inverse_transform(pca_full_data)

print(f"Original shape: {data_reconstructed.shape}")
Resp1_restored = np.zeros((n_storms, n_steps, n_nodes))

for storm_idx in range(n_storms):
    for loc_idx in range(n_nodes):
        start_idx = loc_idx * n_steps
        end_idx = start_idx + n_steps
        Resp1_restored[storm_idx, :, loc_idx] = data_reconstructed[storm_idx, start_idx:end_idx]

print(f"Original shape: {Resp1_restored.shape}")


plt.figure(figsize=(12, 6))
plt.plot(resp_final_dist[node_idx, :, storm_idx], label="Original_Data", linestyle='-')
plt.plot(resp_final_dist_transposed[storm_idx, :, node_idx], label="Original_Data", linestyle='--')
plt.plot(Resp1_restored[storm_idx, :, node_idx], label="Original_Data", linestyle=':')
plt.show()

