import tensorflow as tf

# Define MSE function for custom loss if needed (required if mse was used in the original model)
def mse(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))

# Load the saved autoencoder model
loaded_autoencoder = tf.keras.models.load_model("autoencoder_model_latesttanhtrialnonstd.h5", custom_objects={'mse': mse})

# Display the summary of the loaded model
loaded_autoencoder.summary()
