import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.ticker import FuncFormatter
from scipy.io import loadmat
import h5py

# Load datasets
with h5py.File('clusteredTEST_datasetNACCSupdated.mat', 'r') as f1:
    data1 = np.array(f1['Resp_clust_test'])

mat_file = 'reconstructed_y_hat_softplustrial16.mat'
data2 = loadmat(mat_file)['resptrial2']
data2 = np.transpose(data2, (2, 1, 0))

with h5py.File('NACCS_testDATAPCAwith72compo30000locations.mat', 'r') as f3:
    data3 = np.array(f3['Y_hat'])

# Load centroid data
centroids_df = pd.read_csv('centroids_and_closest_nodes_updated.csv')
latitudes = centroids_df['Node_Latitude'].values
longitudes = centroids_df['Node_Longitude'].values

# Formatter for degrees
def degree_formatter(x, _):
    return f"{x:.1f}\u00b0"

# Set up subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True, sharex=True, sharey=True)

# Plot settings
storm_index = 18

time_index = 130
titles = ["(a) Actual", "(b) GP-AE", "(c) GP-PCA"]
datasets = [data1, data2, data3]

# Longitude and Latitude bounds for the basemap
extent = [-74.2, -71.5, 39.0, 42.8]

# Cap the maximum value (vmax) at 2.5
vmax = 2.5
vmin = 0  # Optionally set a minimum cap

for i, (ax, dataset, title) in enumerate(zip(axs, datasets, titles)):
    plot_data = dataset[:, time_index, storm_index]
    
    # Scatter plot with capped vmax
    sc = ax.scatter(longitudes, latitudes, c=plot_data, cmap='jet', marker='o', s=10, vmin=vmin, vmax=vmax)
    
    # Add basemap with empty attribution
    ctx.add_basemap(
        ax,
        crs="EPSG:4326",
        source=ctx.providers.CartoDB.PositronNoLabels,
        zoom=8,
        attribution=''  # Removes the attribution text
    )
    
    # Adjust plot aesthetics
    ax.set_title(title, fontsize=14)
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    ax.xaxis.set_major_formatter(FuncFormatter(degree_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(degree_formatter))
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.9)

# Add shared colorbar
cbar = fig.colorbar(sc, ax=axs, orientation='vertical', shrink=0.75, aspect=30)
cbar.set_label("Surge (m)", fontsize=14)

# Adjust colorbar ticks to intervals of 0.5 for better granularity
cbar.set_ticks(np.arange(vmin, vmax + 0.5, 0.5))  # Ticks from vmin to vmax at intervals of 0.5

# Add labels
fig.supxlabel("Longitude", fontsize=14)
fig.supylabel("Latitude", fontsize=14)

# Save the figure
plt.savefig("334final_comparison_plot_capped_at_2.5m.svg", format="svg", bbox_inches='tight')

# Show the plot
plt.show()