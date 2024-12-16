
# ####tanh#####################################################################
import numpy as np
import matplotlib.pyplot as plt

# Generate input values (x-axis)
x = np.linspace(-5, 5, 500)  # From -5 to 5 with 500 points

# Compute tanh values
y = np.tanh(x)

# Plot the tanh activation function
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='tanh(x)', color='blue', linewidth=2)

# Add grid, labels, and title
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Horizontal axis
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # Vertical axis
plt.title('Tanh Activation Function', fontsize=16)
plt.xlabel('Input', fontsize=12)
plt.ylabel('Activation Output', fontsize=12)
plt.grid(False)
plt.legend(fontsize=12)

# Set x-axis limits dynamically
plt.xlim(x.min(), x.max())

# Save the plot as an SVG file
plt.savefig("tanh.svg", format="svg", bbox_inches='tight')

# Show the plot
plt.show()


###########sigmoid#############################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # Efficient implementation of sigmoid

# Define the custom sigmoid function
def custom_sigmoid(x):
    return 2 * expit(x) - 1  # expit is the sigmoid function

# Generate input values (x-axis)
x = np.linspace(-10, 10, 500)  # Input range from -10 to 10

# Compute custom sigmoid values
y = custom_sigmoid(x)

# Plot the custom sigmoid function
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Custom Sigmoid', color='blue', linewidth=2)

# Set x-axis limits to the min and max of x
plt.xlim(x.min(), x.max())

# Add grid, labels, and title
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Horizontal axis
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # Vertical axis
plt.title('Custom Sigmoid Function: $2 \\cdot \\text{sigmoid}(x) - 1$', fontsize=16)
plt.xlabel('Input', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.grid(False)
plt.legend(fontsize=12)

# Save the plot as an SVG file
plt.savefig("custom_sigmoid.svg", format="svg", bbox_inches='tight')

# Show the plot
plt.show()

####################Shifted_Selu###############################################

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Define the shifted SELU function
def shifted_selu(x):
    return tf.keras.activations.selu(x) - 10

# Generate input values (x-axis)
x = np.linspace(-10, 10, 500)

# Compute the shifted SELU values
y = shifted_selu(x)

# Plot the shifted SELU function
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Shifted SELU: $\\text{selu}(x) - 10$', color='blue', linewidth=2)

# Add x-axis limits
plt.xlim(x.min(), x.max())

# Add grid, labels, and title
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Horizontal axis
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # Vertical axis
plt.title('Shifted SELU Activation Function', fontsize=16)
plt.xlabel('Input', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.grid(False)
plt.legend(fontsize=12)

# Save the plot as an SVG file
plt.savefig("shifted_selu_plot.svg", format="svg", bbox_inches='tight')

# Display the plot
plt.show()

# Calculate the minimum value of shifted SELU
min_value = shifted_selu(-np.inf).numpy()
print(f"The minimum value of shifted SELU is approximately: {min_value}")


################Shifted_Softplus###############################################

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Define the shifted Softplus function
def shifted_softplus(x):
    return tf.keras.activations.softplus(x) - 10

# Generate input values (x-axis)
x = np.linspace(-10, 10, 500)

# Compute the shifted Softplus values
y = shifted_softplus(x).numpy()  # Convert TensorFlow tensor to NumPy array

# Plot the shifted Softplus function
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Shifted Softplus: $\\text{softplus}(x) - 10$', color='green', linewidth=2)

# Add x and y-axis limits
plt.xlim(x.min(), x.max())
plt.ylim(y.min(), y.max())

# Add grid, labels, and title
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Horizontal axis
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # Vertical axis
plt.title('Shifted Softplus Activation Function', fontsize=16)
plt.xlabel('Input', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.grid(False)
plt.legend(fontsize=12)

# Save the plot as an SVG file
plt.savefig("shifted_softplus_plot.svg", format="svg", bbox_inches='tight')

# Display the plot
plt.show()

# Calculate the minimum value of shifted Softplus
min_value = y.min()  # Use NumPy's min method
print(f"The minimum value of shifted Softplus is approximately: {min_value:.3f}")




