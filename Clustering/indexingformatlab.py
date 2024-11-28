import pandas as pd

# Load the CSV file into a DataFrame
closest_nodes_df = pd.read_csv('centroids_and_closest_nodes1.csv')

# Add 1 to every cell in the 'Node_Index' column
closest_nodes_df['Node_Index'] = closest_nodes_df['Node_Index'] + 1

# Save the updated DataFrame back to a CSV file
closest_nodes_df.to_csv('centroids_and_closest_nodes_updated.csv', index=False)

print("Updated 'Node_Index' column saved to 'centroids_and_closest_nodes_updated.csv'.")