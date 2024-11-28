import h5py
import numpy as np
import pandas as pd

# Load the data from the .mat file using h5py
with h5py.File('NACCS_testDATAPCAwith72compo30000locations.mat', 'r') as test:
    Resp_test = test['Resp_test'][:]
    Y_hat = test['Y_hat'][:]
    param = test['Param_test'][:]

# Debug: Print shapes of loaded arrays
print("Shapes before transposing:")
print("Resp_test shape:", Resp_test.shape)
print("Y_hat shape:", Y_hat.shape)
print("param shape:", param.shape)

# Ensure the dimensions are correctly aligned if needed (transposing might be necessary)
Resp_test = np.transpose(Resp_test, axes=(2, 1, 0))
Y_hat = np.transpose(Y_hat, axes=(2, 1, 0))

# Debug: Print shapes after transposing
print("Shapes after transposing:")
print("Resp_test shape:", Resp_test.shape)
print("Y_hat shape:", Y_hat.shape)

# Define the number of top differences to find
top_n = 100
num_storms = Resp_test.shape[0]

# Initialize an array to store the top locations for each storm
top_locations_array = np.zeros((top_n, num_storms), dtype=int)

# Iterate through each storm and calculate differences
for storm in range(num_storms):
    df = np.zeros(Resp_test.shape[2])  # Assuming third dimension size
    
    # Calculate the sum of absolute differences for each location
    for i in range(Resp_test.shape[2]):
        diff = np.sum(np.abs(np.squeeze(Resp_test[storm, :, i]) - np.squeeze(Y_hat[storm, :, i])))
        df[i] = diff
        # Debug: Print intermediate results
        if i % 1000 == 0:  # Print every 1000 iterations to avoid too much output
            print(f"Processed storm {storm}, location {i}, difference: {diff}")

    # Find the indices of the top 100 largest differences
    ind = np.argpartition(df, -top_n)[-top_n:]  # Get top 'top_n' indices
    ind = ind[np.argsort(df[ind])[::-1]]  # Sort indices by the largest differences

    # Store the top locations for the current storm
    top_locations_array[:, storm] = ind

    # Debug: Print the indices and their corresponding differences
    print(f"Top differences for storm {storm}:")
    for i in range(top_n):
        print(f"Index: {ind[i]}, Difference: {df[ind[i]]}")

# Return the top locations array
print("Top locations array for all storms:")
print(top_locations_array)
print(top_locations_array.shape)

# Count Occurrences and Track Storms
location_counts = np.bincount(top_locations_array.flatten())
location_storms = {}

for storm in range(top_locations_array.shape[1]):
    for location in top_locations_array[:, storm]:
        if location not in location_storms:
            location_storms[location] = set()
        location_storms[location].add(storm)

# Create and Sort DataFrame
location_data = {
    'Location': [],
    'Count': [],
    'Storms': []
}

for location, count in enumerate(location_counts):
    if count > 0:
        location_data['Location'].append(location)
        location_data['Count'].append(count)
        location_data['Storms'].append(sorted(location_storms[location]))

location_counts_df = pd.DataFrame(location_data)

# Sort the DataFrame by the count in descending order
location_counts_df = location_counts_df.sort_values(by='Count', ascending=False)

# Print and Save Results
print(location_counts_df)

# If you want to save the DataFrame to a CSV file, uncomment the following line:
location_counts_df.to_csv('location_countsnew.csv', index=False)
