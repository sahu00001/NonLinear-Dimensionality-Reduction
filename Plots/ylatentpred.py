import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Define MSE function (required to load the model)
def mse(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))

# Load the trained autoencoder model
autoencoder = tf.keras.models.load_model("autoencoder_model_latesttanhtrialnonstd.h5", custom_objects={'mse': mse})

# Extract the decoder part of the model
decoder = tf.keras.Model(inputs=autoencoder.get_layer("dense_2").input, outputs=autoencoder.output)

# Load the latent data from the .mat file
latent_file_path = "latent_Y_Predv2.mat"
with h5py.File(latent_file_path, 'r') as f_latent:
    y_hat_latent = np.array(f_latent['Y_hat_latent'])  # Replace with the correct key if needed

print(f"Latent data shape: {y_hat_latent.shape}")

y_hat_latent_transposed = y_hat_latent.T  # Transpose from (113, 60) to (60, 113)
print(f"Transposed latent data shape: {y_hat_latent_transposed.shape}")

# Ensure latent data shape matches expected input for the decoder
if y_hat_latent_transposed.shape[1] != 113:
    raise ValueError(f"Expected latent dimension of 113, but got {y_hat_latent.shape[1]}")

# Pass the latent data directly to the decoder
reconstructed_from_latent = decoder.predict(y_hat_latent_transposed)

# Save the reconstructed data
reconstructed_output_path = "reconstructed_y_hat_pred.csv"
pd.DataFrame(reconstructed_from_latent).to_csv(reconstructed_output_path, index=False, header=False)

print(f"Reconstructed data shape: {reconstructed_from_latent.shape}")
print(f"Reconstructed data saved to {reconstructed_output_path}")


def plot_time_series_comparison(storm_idx, node_idx):
    # Extract the full time series data for the specified storm and node
    resp1_data = Resp1_restored[storm_idx, :, node_idx]
    resp2_data = Resp2_restored[storm_idx, :, node_idx]
    resp3_data = Resp3_restored[storm_idx, :, node_idx]
    #resp4_data = Resp4_restored[storm_idx, :, node_idx]
    #resp5_data = Resp5_restored[storm_idx, :, node_idx]
    #resp6_data = Resp6_restored[storm_idx, :, node_idx]
    #resp7_data = Resp7_restored[storm_idx, :, node_idx]

    # Generate x-axis labels for timesteps
    timesteps = np.arange(len(resp1_data))  # Assuming timesteps are sequential integers

    # Plot each dataset's time series on the same plot
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, resp1_data, label="Trial14", linestyle='-')
    plt.plot(timesteps, resp2_data, label="Original", linestyle='-')
    plt.plot(timesteps, resp3_data, label="pca", linestyle='-')
    #plt.plot(timesteps, resp4_data, label="Original Data", linestyle='-')
    #plt.plot(timesteps, resp5_data, label="Trial_7", linestyle='-')
    #plt.plot(timesteps, resp6_data, label="Trial_14", linestyle='-')
    #plt.plot(timesteps, resp7_data, label="Trial_111", linestyle='-')

    # Labeling
    plt.title(f"Comparison of Time Series Data for Storm {storm_idx} at Node {node_idx}")
    plt.xlabel("Timesteps")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(False)
    plt.show()
