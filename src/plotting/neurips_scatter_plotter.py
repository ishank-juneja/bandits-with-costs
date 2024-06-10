# Process an entire folder of log files and plot on a single line plot
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import argparse
import pathlib


# Command line input
parser = argparse.ArgumentParser()
parser.add_argument("--log-file-folder", action="store", dest="folder")
parser.add_argument('--algos', type=str, nargs='+',
                    help='Algorithms for regret to be plotted')
parser.add_argument('--save-dir', type=str, help='The directory to save the plots in.')
args = parser.parse_args()

log_folder = args.folder
# Get list of the algorithms for which metrics to be plotted
selected_algos = args.algos
# Number of distinct algorithms used
nalgos = len(selected_algos)
# Color Management
# - - - - - - - - - - - -
# Number of colors should be at least as many as number of LABELS
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
# Choose as many colors as there are algorithms selected_algos
if len(COLORS) < nalgos:
	print("Error: Not enough colors present for plotting\nAdd colors or reduce number of algorithms")
	exit(-1)
else:
	COLORS = COLORS[:nalgos]
# - - - - - - - - - - - -

# Initialize horizon with -1
horizon = -1
nseeds = -1

# Retrieve the number of files to be processed
sorted_files = sorted(pathlib.Path(log_folder).iterdir(), key=lambda x: x.name)
num_files = len(sorted_files)

# Infer the random seed and horizon and ensure their consistency across all the log files
for file_idx, in_file in enumerate(sorted_files):
	# Read in the log file as a pandas dataframe
	bandit_data = pd.read_csv(in_file, sep=",")
	# Infer the horizon as the largest entry in the 'horizon' column
	horizon_new = bandit_data["time-step"].max()
	# Infer the number of distinct random seeds
	# - - - - - - - - - - - -
	nseeds_new = bandit_data["rs"].max() + 1
	# If horizon being set for the first time, set it directly, else check if it is the same as the previous horizon
	if horizon == -1:
		horizon = horizon_new
	elif horizon != horizon_new:
		raise ValueError("Horizon mismatch in log files")

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

	# Plotting
	# - - - - - - - - - - - -
	
	for index, label in enumerate(selected_algos):
		algo_data = bandit_data[bandit_data["algo"] == label]
		x_points_all_runs = algo_data.loc[algo_data["time-step"] == horizon]['qual_reg']
		y_points_all_runs = algo_data.loc[algo_data["time-step"] == horizon]['cost_reg']
		x_points[index, file_idx, :] = x_points_all_runs
		y_points[index, file_idx, :] = y_points_all_runs

# Flatten the x_points and y_points arrays along the last 2 dimensions
x_points = x_points.reshape((nalgos, num_files * nseeds))
y_points = y_points.reshape((nalgos, num_files * nseeds))

# Plot once all the data has been collected together by iterating over all the .csv files
plt.figure(figsize=(10, 10))

for index in range(nalgos):
	plt.scatter(x_points[index, :], y_points[index, :], marker='o', s=100, color=COLORS[index])

# Retrieve tha path for the directory to save the plots in
save_dir = args.save_dir

plt.legend(selected_algos, fontsize='large')  # Increase font size for legend
plt.xlabel("Quality Regret", fontweight="bold", fontsize=14)  # Increase font size for x-axis label
plt.ylabel("Cost Regret", fontweight="bold", fontsize=14)  # Increase font size for y-axis label
# Set font size of ticks
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title("Policy Comparisons", fontweight="bold", fontsize=16)
# - - - - - - - - - - - -

# Save figure
plt.savefig(save_dir + "/neurips_experiment.png", bbox_inches="tight")
plt.close()
