import optuna
import h5py
import gc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from optuna.samplers import RandomSampler

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

# Define the autoencoder model
def create_autoencoder(input_dim, encoding_dim, encoder_neurons, decoder_neurons, learning_rate, optimizer_choice):
    input_layer = Input(shape=(input_dim,))

    # Encoder
    x = input_layer
    for neuron in encoder_neurons:
        x = Dense(
            neuron,
            activation="tanh"
        )(x)

    # Latent space layer
    latent_space = Dense(
        encoding_dim,
        activation="tanh"
    )(x)

    # Decoder
    x = latent_space
    for neuron in decoder_neurons:
        x = Dense(
            neuron,
            activation="tanh"
        )(x)

    # Output layer
    decoded_output = Dense(
        input_dim,
        activation="linear"  # Linear activation for reconstruction
    )(x)

    # Select optimizer based on the trial
    if optimizer_choice == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    # Compile the model
    autoencoder = Model(inputs=input_layer, outputs=decoded_output)
    autoencoder.compile(optimizer=optimizer, loss='mse')

    return autoencoder

# Define the objective function for Optuna
def objective(trial):
    encoding_dim = trial.suggest_int('encoding_dim', 80, 140)  # Latent dimension
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 8, 100)
    epochs = 150

    # Encoder configuration
    num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 5)
    encoder_neurons = [trial.suggest_int(f'encoder_neurons_l{i}', 200, 2500) for i in range(num_encoder_layers)]

    # Decoder configuration (separate from encoder)
    num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 5)
    decoder_neurons = [trial.suggest_int(f'decoder_neurons_l{i}', 100, 2500) for i in range(num_decoder_layers)]

    # Add optimizer choice
    optimizer_choice = trial.suggest_categorical('optimizer', ['adam', 'sgd'])

    # Set up K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    val_losses = []
    for train_index, val_index in kf.split(flattened_data):
        # Split data into training and validation sets for this fold
        X_train, X_val = flattened_data[train_index], flattened_data[val_index]

        # Create a new instance of the autoencoder for each fold
        autoencoder = create_autoencoder(
            input_dim,
            encoding_dim,
            encoder_neurons,
            decoder_neurons,
            learning_rate,
            optimizer_choice
        )

        # Train the model on this fold
        history = autoencoder.fit(
            X_train, X_train,  # Input data is the same as the target data
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            shuffle=True,
            verbose=0  # Suppress verbose output
        )

        # Evaluate the model on the validation set at the end of the fold
        val_loss = autoencoder.evaluate(X_val, X_val, verbose=0)
        val_losses.append(val_loss)

        gc.collect()

    # Return the average validation loss across all folds
    return np.mean(val_losses)

# Set up the Optuna study
study = optuna.create_study(sampler=RandomSampler())
study.optimize(objective, n_trials=100)
df_trials = study.trials_dataframe()
df_trials.to_csv('tanhtrial1updated.csv', index=False)

# Print the best hyperparameters
print(f"Best hyperparameters: {study.best_params}")
print(f"Best value (loss): {study.best_value}")

