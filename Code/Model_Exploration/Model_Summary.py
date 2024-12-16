import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K  


#Define custom sigmoid activation function
# def custom_sigmoid(x):
#     return 2 * tf.keras.activations.sigmoid(x) - 1

# #Register the custom sigmoid activation function
# get_custom_objects().update({'custom_sigmoid': Activation(custom_sigmoid)})

# #Define shifted Softplus activation function
# def shifted_selu(x):
#   return tf.keras.activations.selu(x) - 10

# get_custom_objects().update({'shifted_selu': Activation(shifted_selu)})

# def shifted_softplus(x):
#   return tf.keras.activations.softplus(x) - 10

# get_custom_objects().update({'shifted_softplus': Activation(shifted_softplus)})

# Define RMSE, MSE, and R-squared functions
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def r_squared(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))  # Residual sum of squares
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))  # Total sum of squares
    return 1 - ss_res / (ss_tot + K.epsilon())  # R-squared formula

# Load the saved autoencoder model
loaded_autoencoder = tf.keras.models.load_model(
    "tanhtrial1updatedtrial12.h5",
    custom_objects={
        'mse': mse,
        #'custom_sigmoid': custom_sigmoid,
        #'shifted_softplus': shifted_softplus,
        #'shifted_selu': shifted_selu,
        'rmse': rmse,
        'r_squared': r_squared
    }
)

# Display the summary of the loaded model
loaded_autoencoder.summary()
