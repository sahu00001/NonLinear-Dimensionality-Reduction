import h5py
import gc
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

# Define shifted Softplus activation function
def shifted_softplus(x):
    return tf.keras.activations.softplus(x) - 10

# Define RMSE, MSE, and R-squared functions
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def r_squared(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))  # Residual sum of squares
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))  # Total sum of squares
    return 1 - ss_res / (ss_tot + K.epsilon())  # R-squared formula

# Load and preprocess the data
with h5py.File('clustered_datasetNACCSupdated.mat', 'r') as f_input:
    resp_final_dis = np.array(f_input['Resp_clust'])  # Shape: (3000, 170, 535)

print(f"Original shape: {resp_final_dis.shape}")  # (3000, 170, 535)

# Transpose the data to (535, 170, 3000)
resp_final_dis_transposed = np.transpose(resp_final_dis, (2, 1, 0))
print(f"Transposed shape: {resp_final_dis_transposed.shape}")  # (535, 170, 3000)

# Manually flatten the data to (535, 510000), ensuring the correct order
flattened_data = np.zeros((535, 170 * 3000))
for storm_idx in range(535):
    flattened = []
    for loc_idx in range(3000):
        flattened.extend(resp_final_dis_transposed[storm_idx, :, loc_idx])
    flattened_data[storm_idx, :] = flattened

print(f"Manually flattened shape: {flattened_data.shape}")  # Should be (535, 510000)

input_dim = flattened_data.shape[1]
print("Input dimension:", input_dim)

# Define the autoencoder model with shifted Softplus activation
def create_autoencoder(input_dim, encoding_dim, encoder_neurons, decoder_neurons):
    input_layer = Input(shape=(input_dim,))

    # Build encoder layers
    x = input_layer
    for neuron in encoder_neurons:  # Define encoder structure
        x = Dense(neuron, activation=shifted_softplus)(x)

    # Latent space layer
    latent_space = Dense(encoding_dim, activation=shifted_softplus)(x)

    # Build decoder layers (neurons increase in size in the decoder)
    x = latent_space
    for neuron in decoder_neurons:  # Define decoder structure with increasing neurons
        x = Dense(neuron, activation=shifted_softplus)(x)

    # Output layer
    decoded_output = Dense(input_dim, activation="linear")(x)

    # Create the autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoded_output)
    return autoencoder

# Define parameters manually
encoding_dim = 97  # Manually setting the encoding dimension
encoder_neurons = [1861, 1618, 2143, 329]  # Encoder: decreasing number of neurons
decoder_neurons = [1521, 1286, 291]  # Decoder: increasing number of neurons
epochs = 5000  # Number of epochs
batch_size = 30  # Batch size
learning_rate = 0.00004114187442483321  # Custom learning rate

# Split the data into train and test sets (80-20)
X_train, X_val = train_test_split(flattened_data, test_size=0.2, random_state=42)

# Create the autoencoder
autoencoder = create_autoencoder(input_dim, encoding_dim, encoder_neurons, decoder_neurons)

# Compile the model with RMSE as the loss function and track R-squared
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
autoencoder.compile(optimizer=optimizer, loss=rmse, metrics=[mse, r_squared])

# Define early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)

# Train the autoencoder
history = autoencoder.fit(
    X_train, X_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, X_val),
    shuffle=True,
    verbose=1,
    callbacks=[early_stopping]
)

# Save the trained model
autoencoder.save("softplustrial33.h5")

# Convert metrics to DataFrame for saving
metrics_df = pd.DataFrame({
    "Epoch": range(1, len(history.history['loss']) + 1),
    "Training RMSE": history.history['loss'],
    "Validation RMSE": history.history['val_loss'],
    "Training MSE": history.history['mse'],
    "Validation MSE": history.history['val_mse'],
    "Training R-squared": history.history['r_squared'],
    "Validation R-squared": history.history['val_r_squared']
})

# Save metrics to CSV
output_csv_file = 'softplustrial33.csv'
metrics_df.to_csv(output_csv_file, index=False)
print(f"Training and validation metrics saved to '{output_csv_file}'")

# Final evaluation on the validation set
eval_metrics = autoencoder.evaluate(X_val, X_val, verbose=1)
final_val_rmse = eval_metrics[0]  # RMSE (loss)
final_val_mse = eval_metrics[1]   # MSE
final_val_r2 = eval_metrics[2]    # R-squared

print(f"Final Evaluation on Validation Set - RMSE: {final_val_rmse}, MSE: {final_val_mse}, R-squared: {final_val_r2}")



# Cleanup
gc.collect()
print("Training complete and garbage collected.")

