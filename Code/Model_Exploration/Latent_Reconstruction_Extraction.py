import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K  

#Define custom sigmoid activation function
# def custom_sigmoid(x):
#     return 2 * tf.keras.activations.sigmoid(x) - 1

def shifted_softplus(x):
  return tf.keras.activations.softplus(x) - 10

# Register the custom sigmoid activation function
get_custom_objects().update({'shifted_softplus': Activation(shifted_softplus)})

#Register the custom sigmoid activation function
# get_custom_objects().update({'custom_sigmoid': Activation(custom_sigmoid)})

# def shifted_selu(x):
#   return tf.keras.activations.selu(x) - 10

# get_custom_objects().update({'shifted_selu': Activation(shifted_selu)})


# Define RMSE, MSE, and R-squared functions
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def r_squared(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))  # Residual sum of squares
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))  # Total sum of squares
    return 1 - ss_res / (ss_tot + K.epsilon())  # R-squared formula


# Load the trained autoencoder model and retrieve the encoder
autoencoder = tf.keras.models.load_model("tanhtrial1updatedtrial12.h5", custom_objects={'mse': mse, 'rmse': rmse,'r_squared': r_squared})
encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("dense_3").output)

# Load and preprocess the data
with h5py.File('clusteredTEST_datasetNACCSupdated.mat', 'r') as f_input:
    resp_final_dis = np.array(f_input['Resp_clust_test'])  # Shape: (3000, 170, 535)

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
pd.DataFrame(reconstructed_data).to_csv("reconstructedAE60tanhtrial1updatedtrial12.csv", index=False, header=False)
print(f"Reconstructed data shape: {reconstructed_data.shape}")

# Use the encoder to get the latent representation
latent_output = encoder.predict(flattened_data)
# pd.DataFrame(latent_output).to_csv("535latentselutrial5.csv", index=False, header=False)
print(f"Latent output shape: {latent_output.shape}")

print("Reconstructed data and latent representation saved successfully.")
