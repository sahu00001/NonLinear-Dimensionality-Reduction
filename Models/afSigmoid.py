from sklearn.model_selection import train_test_split
import h5py
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import get_custom_objects

# Define custom sigmoid activation function
def custom_sigmoid(x):
    return 2 * tf.keras.activations.sigmoid(x) - 1

# Register the custom sigmoid activation function
get_custom_objects().update({'custom_sigmoid': Activation(custom_sigmoid)})

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
with h5py.File('clustered_datasetNACCS.mat', 'r') as f_input:
    resp_final_dis = np.array(f_input['Resp_clust'])  # Shape: (3000, 170, 535)

resp_final_dis_transposed = np.transpose(resp_final_dis, (2, 1, 0))

flattened_data = np.zeros((535, 170 * 3000))
for storm_idx in range(535):
    flattened = []
    for loc_idx in range(3000):
        flattened.extend(resp_final_dis_transposed[storm_idx, :, loc_idx])
    flattened_data[storm_idx, :] = flattened


input_dim = flattened_data.shape[1]

# Define the autoencoder model
def create_autoencoder(input_dim, encoding_dim, encoder_neurons, decoder_neurons):
    input_layer = Input(shape=(input_dim,))
    x = input_layer
    for neuron in encoder_neurons:
        x = Dense(neuron, activation=custom_sigmoid)(x)  # Use custom sigmoid
    latent_space = Dense(encoding_dim, activation=custom_sigmoid)(x)  # Latent space with custom sigmoid
    x = latent_space
    for neuron in decoder_neurons:
        x = Dense(neuron, activation=custom_sigmoid)(x)  # Use custom sigmoid
    decoded_output = Dense(input_dim, activation="linear")(x)
    autoencoder = Model(inputs=input_layer, outputs=decoded_output)
    return autoencoder

# Parameters
encoding_dim = 106
encoder_neurons = [1509]
decoder_neurons = [2471]
epochs = 5000
batch_size = 81
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Split data into training (80%) and testing (20%) sets
X_train, X_test = train_test_split(flattened_data, test_size=0.2, random_state=42)

# Create the autoencoder
autoencoder = create_autoencoder(input_dim, encoding_dim, encoder_neurons, decoder_neurons)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00002474005272779104)
autoencoder.compile(optimizer=optimizer, loss=rmse, metrics=[mse, r_squared])

# Train the autoencoder on the training set
history = autoencoder.fit(
    X_train, X_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, X_test),
    shuffle=True,
    verbose=1,
    callbacks=[early_stopping]
)

# Evaluate the model on the test set
eval_metrics = autoencoder.evaluate(X_test, X_test, verbose=1)
final_test_rmse = eval_metrics[0]
final_test_mse = eval_metrics[1]
final_test_r2 = eval_metrics[2]
print(f"Test Set - RMSE: {final_test_rmse}, MSE: {final_test_mse}, R-squared: {final_test_r2}")


model_save_path = '5000sigmoidtrial77.h5'
autoencoder.save(model_save_path)
print(f"Trained model saved to '{model_save_path}'")


# Convert history to DataFrame and save
all_metrics = [
    [epoch + 1, history.history['loss'][epoch], history.history['val_loss'][epoch],
     history.history['mse'][epoch], history.history['val_mse'][epoch],
     history.history['r_squared'][epoch], history.history['val_r_squared'][epoch]]
    for epoch in range(len(history.history['loss']))
]
metrics_df = pd.DataFrame(all_metrics, columns=["Epoch", "Training RMSE", "Validation RMSE", "Training MSE", "Validation MSE", "Training R-squared", "Validation R-squared"])
output_csv_file = '5000sigmoidtrial77.csv'
metrics_df.to_csv(output_csv_file, index=False)

print(f"Training and validation losses and metrics saved to '{output_csv_file}'")

