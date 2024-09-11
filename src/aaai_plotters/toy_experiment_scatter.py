# Plotter for the experiment where we sweep the return of one arm
#  while keeping the other arms in the instance fixed
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import argparse
import pathlib
from matplotlib.lines import Line2D

# Command line input
parser = argparse.ArgumentParser()
parser.add_argument("--folder", action="store", dest="folder")
parser.add_argument('--algos', type=str, nargs='+', help='Algorithms for regret to be plotted')
parser.add_argument('--save-dir', type=str, help='The directory to save the plots in.')
args = parser.parse_args()

log_folder = args.folder
selected_algos = args.algos

# Map algo names used in the log files
#  to algo names used in the code
custom_algo_names = {
    'cs-pe': 'pe-cs',
    'cs-ucb': 'ucb-cs',
    'cs-ts': 'ts-cs',
    'cs-etc': 'etc-cs'
}

# Use difference scatter plot marker style for every algorithm
marker_styles = ['o', 's', '^', '*']
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

nalgos = len(selected_algos)

# Initialize nseeds with -1
nseeds = -1

# Retrieve the number of files to be processed
sorted_files = sorted(pathlib.Path(log_folder).iterdir(), key=lambda x: x.name)
num_files = len(sorted_files)

# Infer the random seed and ensure its consistency across all the log files
for file_idx, in_file in enumerate(sorted_files):
    # Read in the log file as a pandas dataframe
    bandit_data = pd.read_csv(in_file, sep=",")
    # Infer the number of distinct random seeds
    # - - - - - - - - - - - -
    nseeds_new = bandit_data["rs"].max() + 1
    # Enforce the same number of seeds across all log files
    if nseeds == -1:
        nseeds = nseeds_new
    elif nseeds != nseeds_new:
        raise ValueError("Number of seeds mismatch in log files")

# Init numpy arrays to hold the scatter plot data
x_points = np.zeros((nalgos, num_files, nseeds))
y_points = np.zeros((nalgos, num_files, nseeds))

# Iterate over all the .csv files present in this folder
for file_idx, in_file in enumerate(sorted_files):
    # Data Reading and Preprocessing
    # - - - - - - - - - - - -
    # Read in the log file as a pandas dataframe
    bandit_data = pd.read_csv(in_file, sep=",")

    # Identify the horizon for the current file
    horizon = bandit_data["time-step"].max()
    # Plotting
    # - - - - - - - - - - - -

    for index, label in enumerate(selected_algos):
        algo_data = bandit_data[bandit_data["algo"] == label]
        x_points_all_runs = algo_data.loc[algo_data["time-step"] == horizon]['qual_reg']
        y_points_all_runs = algo_data.loc[algo_data["time-step"] == horizon]['cost_reg']
        x_points[index, file_idx, :] = x_points_all_runs / horizon * 10**5
        y_points[index, file_idx, :] = y_points_all_runs / horizon * 10**5

# Fix x-tick positions to hold the labels for the bandit instance
#  with the varying arm
# - - - - - - - - - - - - - -
# 12 equispaced points between 0 and 1 (inclusive)
cost_xs = np.linspace(0, 1, 12) # 12 cost regret experiments
cost_xs_markers = np.linspace(0.6, 0.93, 12) # The value of the varying first arm for the instance family
# Convert the NumPy array to a list of strings, formatted to 2 decimal places
cost_xs_markers_str = [f'{x:.2f}' for x in cost_xs_markers]
# 8 equi-spaced points between 0 and 1 (inclusive)
qual_xs = np.linspace(0, 1, 8)  # 8 quality regret experiments
qual_xs_markers = np.linspace(0.01, 0.15, 8)  # The value of the varying second arm for the instance family
qual_xs_markers_str = [f'{x:.2f}' for x in qual_xs_markers]
# - - - - - - - - - - - - - -

# Sample data for the first scatter plot of size 12
y1 = np.random.rand(12)
# Sample data for the second scatter plot
y2 = np.random.rand(8)

# Create a figure and a set of subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 10))  # 2 rows, 1 column, figure size 8x10 inches

# Scatter plot on the first subplot
axs[0].scatter(cost_xs, y1, color='blue')  # You can change the color
axs[0].set_title('Scatter Plot 1')
axs[0].set_xlabel('x values')
axs[0].set_ylabel('y values')
axs[0].set_xticks(cost_xs)
axs[0].set_xticklabels(cost_xs_markers_str)  # Setting alphabetical x-tick labels

# Scatter plot on the second subplot
axs[1].scatter(qual_xs, y2, color='green')  # You can change the color
axs[1].set_title('Scatter Plot 2')
axs[1].set_xlabel('x values')
axs[1].set_ylabel('y values')
axs[1].set_xticks(qual_xs)
axs[1].set_xticklabels(qual_xs_markers_str)  # Setting alphabetical x-tick labels

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
