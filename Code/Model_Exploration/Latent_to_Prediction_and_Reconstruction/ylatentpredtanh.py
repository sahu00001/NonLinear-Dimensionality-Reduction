import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K  
##Defining errors for calculation
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def r_squared(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))  # Residual sum of squares
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))  # Total sum of squares
    return 1 - ss_res / (ss_tot + K.epsilon())  # R-squared formula

# Model Loading
autoencoder = tf.keras.models.load_model("tanhtrial3updatedtrial33.h5", custom_objects={ 'mse': mse,'rmse': rmse,'r_squared': r_squared,})

# Extractingdata from decoder
decoder = tf.keras.Model(inputs=autoencoder.get_layer("dense_5").input, outputs=autoencoder.output)

# Load the latent data 
latent_file_path = "NACCS_tanhtrial1updatedtrial33for3000locations.mat"
with h5py.File(latent_file_path, 'r') as f_latent:
    y_hat_latent = np.array(f_latent['Y_hat_latent'])  # Replace with the correct key if needed

print(f"Latent data shape: {y_hat_latent.shape}")

y_hat_latent_transposed = y_hat_latent.T  # Transpose from (113, 60) to (60, 113)
print(f"Transposed latent data shape: {y_hat_latent_transposed.shape}")

# chceking the latent shape matches the data inputed
if y_hat_latent_transposed.shape[1] != 103:
    raise ValueError(f"Expected latent dimension of 118, but got {y_hat_latent.shape[1]}")

# Passing latent directly to the decoder
reconstructed_from_latent = decoder.predict(y_hat_latent_transposed)

# Save the reconstructed data
reconstructed_output_path = "reconstructed_y_hat_tanhtrial3updatedtrial33.csv"
pd.DataFrame(reconstructed_from_latent).to_csv(reconstructed_output_path, index=False, header=False)

print(f"Reconstructed data shape: {reconstructed_from_latent.shape}")
print(f"Reconstructed data saved to {reconstructed_output_path}")



