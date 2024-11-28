import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load and preprocess the data
with h5py.File('clusteredTEST_datasetNACCS.mat', 'r') as f_input:
    resp_final_dis = np.array(f_input['Resp_test_clust'])  # Shape: (3000, 170, 535)

# Transpose and flatten data
resp_final_dis_transposed = np.transpose(resp_final_dis, (2, 1, 0))  # New shape: (535, 170, 3000)
flattened_data = np.zeros((60, 170 * 3000))
for storm_idx in range(60):
    flattened = []
    for loc_idx in range(3000):
        flattened.extend(resp_final_dis_transposed[storm_idx, :, loc_idx])
    flattened_data[storm_idx, :] = flattened

# Standardize the flattened data
scaler = StandardScaler()
manual_standardized_data = scaler.fit_transform(flattened_data)



# Apply PCA for full data
pca_full = PCA()
pca_full_data = pca_full.fit_transform(manual_standardized_data)

# Reconstruct data from principal components
data_reconstructed = pca_full.inverse_transform(pca_full_data)
data_reconstructed_original_scale = scaler.inverse_transform(data_reconstructed)

# Calculate reconstruction error metrics
mse = mean_squared_error(flattened_data, data_reconstructed_original_scale)
mae = mean_absolute_error(flattened_data, data_reconstructed_original_scale)
r2 = r2_score(flattened_data, data_reconstructed_original_scale)

print(f"Reconstruction Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-Squared (R²): {r2}")

# Save the reconstructed data as a CSV file with header = 0
np.savetxt("reconstructed_datapython.csv", data_reconstructed_original_scale, delimiter=",")

# Plot original vs reconstructed data for row 4 (5th row, index 4)
row_idx = 4  # Row index for row 4
plt.figure(figsize=(10, 5))

# Plot the original and reconstructed data for row 4
plt.plot(flattened_data[row_idx], label='Original Data', color='blue', linestyle='-', linewidth=1)
plt.plot(data_reconstructed_original_scale[row_idx], label='Reconstructed Data', color='red', linestyle='--', linewidth=1)

plt.legend()
plt.title(f'Original vs Reconstructed Data (Row {row_idx})')
plt.xlabel('Feature Index')
plt.ylabel('Value')
plt.show()




#pca_full_df = pd.DataFrame(pca_full_data)
#pca_full_df.to_csv("pca_full_data.csv", index=False, header = False)

#print(f"Shape of full PCA transformed data: {pca_full_data.shape}")



#pca_216 = PCA(n_components=216)
#pca_216_data = pca_216.fit_transform(manual_standardized_data)


#pca_216_df = pd.DataFrame(pca_216_data)
#pca_216_df.to_csv("pca_216_data.csv", index=False, header = False)

#print(f"Shape of PCA transformed data with 216 components: {pca_216_data.shape}")


#reconstructed_216 = pca_216.inverse_transform(pca_216_data)
#reconstructed_216_unscaled = scaler.inverse_transform(reconstructed_216)
#reconstructed_216_df = pd.DataFrame(reconstructed_216_unscaled)
#reconstructed_216_df.to_csv("reconstructed_216_data.csv", index=False, header = False)
#
#pca_110 = PCA(n_components=110)

#pca_110_data = pca_110.fit_transform(manual_standardized_data)

#pca_110_df = pd.DataFrame(pca_110_data)
#pca_110_df.to_csv("pca_110_data.csv", index=False, header = False)

#print(f"Shape of PCA transformed data with 110 components: {pca_110_data.shape}")


#reconstructed_110 = pca_110.inverse_transform(pca_110_data)
#reconstructed_110_unscaled = scaler.inverse_transform(reconstructed_110)
#reconstructed_110_df = pd.DataFrame(reconstructed_110_unscaled)
#reconstructed_110_df.to_csv("reconstructed_110_data.csv", index=False,header = False)

#print("Reconstructed data saved for both 216 and 110 components.")