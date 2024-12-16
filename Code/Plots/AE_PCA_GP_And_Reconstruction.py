
#######################AE_PCA_GP#########################################
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Parameters
n_storms = 60  # Number of storms
n_steps = 170  # Number of time steps
n_nodes = 3000  # Number of nodes
trial_files = [  # List of trial file paths
    "reconstructed_y_hat_sigmoidtria1updatedtrial16.csv",
    "reconstructed_y_hat_softplustrial33.csv",
    "reconstructed_y_hat_tanhtrial12updatedtrial12.csv"
]

# Load disp_vect.mat and extract the displacement vector
disp_vect_data = loadmat('disp_vect.mat')  # Adjust the key name based on the file's structure
disp_vect = disp_vect_data['cand_dis'].flatten()  # Assuming 'cand_dis' contains the vector
print(f"Displacement vector loaded with shape: {disp_vect.shape}")  # (170,)

# Load resp3_data
h5_file_path = "NACCS_testDATAPCAwith72compo30000locations.mat"  # Replace with your file path
with h5py.File(h5_file_path, 'r') as f_input:
    resp3_data_raw = np.array(f_input['Y_hat'])
    resp3_data = np.transpose(resp3_data_raw, (2, 1, 0))
    
print(f"Original data (resp3_data_raw) shape: {resp3_data.shape}")

# Load the Resp_test_clust data (original data)
h5_file_path = "clusteredTEST_datasetNACCSupdated.mat"  # Replace with your file path
with h5py.File(h5_file_path, 'r') as f_input:
    resp4_data_raw = np.array(f_input['Resp_clust_test'])  # Original shape: (3000, 170, 535)
    resp4_data = np.transpose(resp4_data_raw, (2, 1, 0))  # Transposed to (535, 170, 3000)

print(f"Original data (resp4_data) shape: {resp4_data.shape}")

# Initialize storage for restored data from all trials
restored_data_trials = []

# Process each trial
for trial_idx, file_path in enumerate(trial_files):
    print(f"Processing Trial {trial_idx + 1}: {file_path}")
    
    # Load the reconstructed data
    data_reconstructed = pd.read_csv(file_path, header=None).values  # Shape: (60, 510000)
    print(f"Loaded reconstructed data shape: {data_reconstructed.shape}")
    
    # Restore the shape for the trial
    restored = np.zeros((n_storms, n_steps, n_nodes))
    for storm_idx in range(n_storms):
        for loc_idx in range(n_nodes):
            start_idx = loc_idx * n_steps
            end_idx = start_idx + n_steps
            restored[storm_idx, :, loc_idx] = data_reconstructed[storm_idx, start_idx:end_idx]
    
    print(f"Restored shape for Trial {trial_idx + 1}: {restored.shape}")
    restored_data_trials.append(restored)

# Save the restored data and original data for future use
output_file = "restored_data_and_original.npz"
np.savez(output_file, original=resp4_data, restored_trials=restored_data_trials)
print(f"Data saved to {output_file}")

# Function to plot the time series comparison
def plot_time_series_comparison_all(storm_idx, node_idx):
    # Extract original data for the specified storm and node
    original_data = resp4_data[storm_idx, :, node_idx]
    resp3_data_node = resp3_data[storm_idx, :, node_idx]
    
    # Plot the original data
    plt.figure(figsize=(12, 8))
    plt.plot(disp_vect, original_data, label="Original Data", linestyle='--', color='black')
    
    # Plot resp3_data
    plt.plot(disp_vect, resp3_data_node, label="PCA_GP", linestyle=':', color='purple')
    
    # Plot each trial's reconstructed data
    for trial_idx, restored in enumerate(restored_data_trials):
        trial_data = restored[storm_idx, :, node_idx]
        trial_name = trial_files[trial_idx].replace("reconstructed_y_hat_", "").replace(".csv", "")
        plt.plot(disp_vect, trial_data, label=trial_name)
        
    plt.xlim(disp_vect.min(), disp_vect.max())
    
    # Labeling
    plt.title(f"Comparison of Time Series Data for Storm {storm_idx} at Node {node_idx}")
    plt.xlabel("Displacement (km)")  # Using disp_vect for x-axis
    plt.ylabel("Values")
    plt.legend()
    plt.grid(False)
    output_filename = f"storm_{storm_idx}_node_{node_idx}.svg"
    plt.savefig(output_filename, format="svg", bbox_inches="tight")
    print(f"Plot saved as: {output_filename}")
    
    plt.show()
    plt.close()

# Example usage
plot_time_series_comparison_all(storm_idx=0, node_idx=0)

#######################AE_PCA_Reconstruction########################################

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Parameters
n_storms = 60  # Number of storms
n_steps = 170  # Number of time steps
n_nodes = 3000  # Number of nodes
trial_files = [  # List of trial file paths
    "reconstructedAE60sigmoidtria3updatedtrial16.csv",
    "reconstructedAE60softplustrial33.csv",
    "reconstructedAE60tanhtrial1updatedtrial12.csv"
]

# Load disp_vect.mat and extract the displacement vector
disp_vect_data = loadmat('disp_vect.mat')  # Adjust the key name based on the file's structure
disp_vect = disp_vect_data['cand_dis'].flatten()  # Assuming 'cand_dis' contains the vector
print(f"Displacement vector loaded with shape: {disp_vect.shape}")  # (170,)

# Load resp3_data
h5_file_path = "Recon_Respfor60stormsamepercetage_97of_retained_info.mat"  # Replace with your file path
with h5py.File(h5_file_path, 'r') as f_input:
    resp3_data_raw = np.array(f_input['Recon_Resp'])
    print(f"Original data (resp3_data_raw) shape: {resp3_data_raw.shape}")
    resp3_data = np.transpose(resp3_data_raw, (2, 0, 1))
    
print(f"Original data (resp3_data_raw) shape: {resp3_data.shape}")

# Load the Resp_test_clust data (original data)
h5_file_path = "clusteredTEST_datasetNACCSupdated.mat"  # Replace with your file path
with h5py.File(h5_file_path, 'r') as f_input:
    resp4_data_raw = np.array(f_input['Resp_clust_test'])  # Original shape: (3000, 170, 535)
    resp4_data = np.transpose(resp4_data_raw, (2, 1, 0))  # Transposed to (535, 170, 3000)

print(f"Original data (resp4_data) shape: {resp4_data.shape}")

# Initialize storage for restored data from all trials
restored_data_trials = []

# Process each trial
for trial_idx, file_path in enumerate(trial_files):
    print(f"Processing Trial {trial_idx + 1}: {file_path}")
    
    # Load the reconstructed data
    data_reconstructed = pd.read_csv(file_path, header=None).values  # Shape: (60, 510000)
    print(f"Loaded reconstructed data shape: {data_reconstructed.shape}")
    
    # Restore the shape for the trial
    restored = np.zeros((n_storms, n_steps, n_nodes))
    for storm_idx in range(n_storms):
        for loc_idx in range(n_nodes):
            start_idx = loc_idx * n_steps
            end_idx = start_idx + n_steps
            restored[storm_idx, :, loc_idx] = data_reconstructed[storm_idx, start_idx:end_idx]
    
    print(f"Restored shape for Trial {trial_idx + 1}: {restored.shape}")
    restored_data_trials.append(restored)

# Save the restored data and original data for future use
output_file = "restored_data_and_original.npz"
np.savez(output_file, original=resp4_data, restored_trials=restored_data_trials)
print(f"Data saved to {output_file}")

# Function to plot the time series comparison
def plot_time_series_comparison_all(storm_idx, node_idx):
    # Extract original data for the specified storm and node
    original_data = resp4_data[storm_idx, :, node_idx]
    resp3_data_node = resp3_data[storm_idx, :, node_idx]
    
    # Plot the original data
    plt.figure(figsize=(12, 8))
    plt.plot(disp_vect, original_data, label="Original Data", linestyle='--', color='black')
    
    # Plot resp3_data
    plt.plot(disp_vect, resp3_data_node, label="PCA_Recon", linestyle=':', color='purple')
    
    # Plot each trial's reconstructed data
    for trial_idx, restored in enumerate(restored_data_trials):
        trial_data = restored[storm_idx, :, node_idx]
        trial_name = trial_files[trial_idx].replace("AE_reconstruction", "").replace(".csv", "")
        plt.plot(disp_vect, trial_data, label=trial_name)
        
    plt.xlim(disp_vect.min(), disp_vect.max())
    
    # Labeling
    plt.title(f"Comparison of Time Series Data for Storm {storm_idx} at Node {node_idx}")
    plt.xlabel("Displacement (km)")  # Using disp_vect for x-axis
    plt.ylabel("Values")
    plt.legend()
    plt.grid(False)
    output_filename = f"storm_{storm_idx}_node_{node_idx}.svg"
    plt.savefig(output_filename, format="svg", bbox_inches="tight")
    print(f"Plot saved as: {output_filename}")
    
    plt.show()
    plt.close()

# Example usage
plot_time_series_comparison_all(storm_idx=0, node_idx=0)


#####################all_softplus_with_original_GP_reconstructed###################################
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# Parameters
n_storms = 60   # Number of storms
n_steps = 170   # Number of time steps
n_nodes = 3000  # Number of nodes

# Load the data from the .mat file using h5py
with h5py.File('NACCS_testDATAPCAwith72compo30000locations.mat', 'r') as test:
    Resp_test = test['Resp_test'][:]
    Y_hat = test['Y_hat'][:]
    param = test['Param_test'][:]

# Print shapes before transposing (for debug)
print("Shapes before transposing:")
print("Resp_test shape:", Resp_test.shape)
print("Y_hat shape:", Y_hat.shape)
print("Param_test shape:", param.shape)

# Transpose to get desired shape (60, 170, 3000)
Resp_test = np.transpose(Resp_test, axes=(2, 1, 0))
Y_hat = np.transpose(Y_hat, axes=(2, 1, 0))

# Print shapes after transposing (for debug)
print("Shapes after transposing:")
print("Resp_test shape:", Resp_test.shape)  # (60, 170, 3000)
print("Y_hat shape:", Y_hat.shape)          # (60, 170, 3000)

# Function to load a reconstructed trial file and reshape it into a 3D array
def load_reconstructed_trial(file_path, n_storms, n_steps, n_nodes):
    data_reconstructed = pd.read_csv(file_path, header=None).values  # Shape: (60, 510000)
    print(f"Loaded reconstructed data shape for {file_path}:", data_reconstructed.shape)
    
    resptrial = np.zeros((n_storms, n_steps, n_nodes))
    for storm_idx in range(n_storms):
        for loc_idx in range(n_nodes):
            start_idx = loc_idx * n_steps
            end_idx = start_idx + n_steps
            resptrial[storm_idx, :, loc_idx] = data_reconstructed[storm_idx, start_idx:end_idx]
            
    print(f"{file_path} reshaped to:", resptrial.shape)
    return resptrial

# Load the three reconstructed trials
resptrial16 = load_reconstructed_trial("reconstructed_y_hat_softplustrial16.csv", n_storms, n_steps, n_nodes)
resptrial21 = load_reconstructed_trial("reconstructed_y_hat_softplustrial21.csv", n_storms, n_steps, n_nodes)
resptrial33 = load_reconstructed_trial("reconstructed_y_hat_softplustrial33.csv", n_storms, n_steps, n_nodes)

def plot_time_series_comparisonsoftplus(storm_idx, node_idx):
    """
    Plot the time series comparison for a given storm and node.
    Shows:
    - Original Data (Resp_test)
    - Y_hat
    - resptrial16 (Reconstructed from trial16 file)
    - resptrial22 (Reconstructed from trial22 file)
    - resptrial33 (Reconstructed from trial33 file)
    """
    # Extract the data for the chosen storm and node
    original_data = Resp_test[storm_idx, :, node_idx]
    y_hat_data = Y_hat[storm_idx, :, node_idx]
    resptrial16_data = resptrial16[storm_idx, :, node_idx]
    resptrial21_data = resptrial21[storm_idx, :, node_idx]
    resptrial33_data = resptrial33[storm_idx, :, node_idx]

    # Generate x-axis labels for timesteps
    timesteps = np.arange(n_steps)

    # Plot the five lines
    plt.figure(figsize=(12, 8))
    plt.plot(timesteps, original_data, label="Original Data (Resp_test)", linestyle='--', color='black')
    plt.plot(timesteps, y_hat_data, label="Y_hat", linestyle=':', color='blue')
    plt.plot(timesteps, resptrial16_data, label="resptrial2 (Trial16)", linestyle='-', color='red')
    plt.plot(timesteps, resptrial21_data, label="resptrial3 (Trial22)", linestyle='-.', color='green')
    plt.plot(timesteps, resptrial33_data, label="resptrial4 (Trial33)", linestyle='-', color='purple')

    plt.title(f"Comparison of Time Series for Storm {storm_idx} at Node {node_idx}")
    plt.xlabel("Timesteps")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage: Plot the comparison for storm 15 at node 1977
plot_time_series_comparisonsoftplus(storm_idx=15, node_idx=1977)




#####################################alltanhtrials_GP_reconstructed_with_original#############################
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# Parameters
n_storms = 60   # Number of storms
n_steps = 170   # Number of time steps
n_nodes = 3000  # Number of nodes

# Load the data from the .mat file using h5py
with h5py.File('NACCS_testDATAPCAwith72compo30000locations.mat', 'r') as test:
    Resp_test = test['Resp_test'][:]
    Y_hat = test['Y_hat'][:]
    param = test['Param_test'][:]

# Print shapes before transposing (for debug)
print("Shapes before transposing:")
print("Resp_test shape:", Resp_test.shape)
print("Y_hat shape:", Y_hat.shape)
print("Param_test shape:", param.shape)

# Transpose to get desired shape (60, 170, 3000)
Resp_test = np.transpose(Resp_test, axes=(2, 1, 0))
Y_hat = np.transpose(Y_hat, axes=(2, 1, 0))

# Print shapes after transposing (for debug)
print("Shapes after transposing:")
print("Resp_test shape:", Resp_test.shape)  # (60, 170, 3000)
print("Y_hat shape:", Y_hat.shape)          # (60, 170, 3000)

# Function to load a reconstructed trial file and reshape it into a 3D array
def load_reconstructed_trial(file_path, n_storms, n_steps, n_nodes):
    data_reconstructed = pd.read_csv(file_path, header=None).values  # Shape: (60, 510000)
    print(f"Loaded reconstructed data shape for {file_path}:", data_reconstructed.shape)
    
    resptrial = np.zeros((n_storms, n_steps, n_nodes))
    for storm_idx in range(n_storms):
        for loc_idx in range(n_nodes):
            start_idx = loc_idx * n_steps
            end_idx = start_idx + n_steps
            resptrial[storm_idx, :, loc_idx] = data_reconstructed[storm_idx, start_idx:end_idx]
            
    print(f"{file_path} reshaped to:", resptrial.shape)
    return resptrial

# Load all reconstructed trials
resptrial1 = load_reconstructed_trial("reconstructed_y_hat_tanhtrial1updatedtrial3.csv", n_storms, n_steps, n_nodes)
resptrial2 = load_reconstructed_trial("reconstructed_y_hat_tanhtrial1updatedtrial25.csv", n_storms, n_steps, n_nodes)
resptrial3 = load_reconstructed_trial("reconstructed_y_hat_tanhtrial1updatedtrial36.csv", n_storms, n_steps, n_nodes)
resptrial4 = load_reconstructed_trial("reconstructed_y_hat_tanhtrial2updatedtrial21.csv", n_storms, n_steps, n_nodes)
resptrial5 = load_reconstructed_trial("reconstructed_y_hat_tanhtrial3updatedtrial33.csv", n_storms, n_steps, n_nodes)
resptrial6 = load_reconstructed_trial("reconstructed_y_hat_tanhtrial12updatedtrial12.csv", n_storms, n_steps, n_nodes)

# Define a function to plot time series comparison
def plot_time_series_comparison(storm_idx, node_idx):
    """
    Plot the time series comparison for a given storm and node.
    Shows:
    - Original Data (Resp_test)
    - Y_hat
    - All reconstructed trials
    """
    # Extract the data for the chosen storm and node
    original_data = Resp_test[storm_idx, :, node_idx]
    y_hat_data = Y_hat[storm_idx, :, node_idx]
    resptrial1_data = resptrial1[storm_idx, :, node_idx]
    resptrial2_data = resptrial2[storm_idx, :, node_idx]
    resptrial3_data = resptrial3[storm_idx, :, node_idx]
    resptrial4_data = resptrial4[storm_idx, :, node_idx]
    resptrial5_data = resptrial5[storm_idx, :, node_idx]
    resptrial6_data = resptrial6[storm_idx, :, node_idx]

    # Generate x-axis labels for timesteps
    timesteps = np.arange(n_steps)

    # Plot the lines
    plt.figure(figsize=(12, 8))
    plt.plot(timesteps, original_data, label="Original Data (Resp_test)", linestyle='--', color='black')
    plt.plot(timesteps, y_hat_data, label="Y_hat", linestyle=':', color='blue')
    plt.plot(timesteps, resptrial1_data, label="resptrial1 (Trial1Updated3)", linestyle='-', color='red')
    plt.plot(timesteps, resptrial2_data, label="resptrial2 (Trial1Updated25)", linestyle='-.', color='green')
    plt.plot(timesteps, resptrial3_data, label="resptrial3 (Trial1Updated36)", linestyle='-', color='purple')
    plt.plot(timesteps, resptrial4_data, label="resptrial4 (Trial2Updated21)", linestyle='-.', color='orange')
    plt.plot(timesteps, resptrial5_data, label="resptrial5 (Trial3Updated33)", linestyle='-', color='brown')
    plt.plot(timesteps, resptrial6_data, label="resptrial6 (Trial12Updated12)", linestyle=':', color='pink')

    plt.title(f"Comparison of Time Series for Storm {storm_idx} at Node {node_idx}")
    plt.xlabel("Timesteps")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage: Plot the comparison for storm 15 at node 1977
plot_time_series_comparison(storm_idx=15, node_idx=1977)

