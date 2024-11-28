import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Define MSE function (required to load the model)
def mse(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))

# Load the trained autoencoder model and retrieve the encoder
autoencoder = tf.keras.models.load_model("autoencoder_model_latesttanhtrialnonstd.h5", custom_objects={'mse': mse})
encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("dense_1").output)

# Load and preprocess the data
with h5py.File('clusteredTEST_datasetNACCS.mat', 'r') as f_input:
    resp_final_dis = np.array(f_input['Resp_test_clust'])  # Shape: (3000, 170, 535)

print(f"Original shape: {resp_final_dis.shape}")  # (3000, 170, 535)

# Transpose and flatten the data to (535, 510000)
resp_final_dis_transposed = np.transpose(resp_final_dis, (2, 1, 0))
print(f"Transposed shape: {resp_final_dis_transposed.shape}")

flattened_data = np.zeros((60, 170 * 3000))
for storm_idx in range(60):
    flattened = []
    for loc_idx in range(3000):  # Stack time steps for all locations
        flattened.extend(resp_final_dis_transposed[storm_idx, :, loc_idx])
    flattened_data[storm_idx, :] = flattened

print(f"Flattened shape: {flattened_data.shape}")  # Should be (535, 510000)

# Use the autoencoder to get the reconstructed output
reconstructed_data = autoencoder.predict(flattened_data)
pd.DataFrame(reconstructed_data).to_csv("reconstructed_tanh60new.csv", index=False, header=False)
print(f"Reconstructed data shape: {reconstructed_data.shape}")

# Use the encoder to get the latent representation
latent_output = encoder.predict(flattened_data)
pd.DataFrame(latent_output).to_csv("latent_tanh60.csv", index=False, header=False)
print(f"Latent output shape: {latent_output.shape}")

print("Reconstructed data and latent representation saved successfully.")
