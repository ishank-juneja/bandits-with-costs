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
parser.add_argument('--metric', type=str, choices=['reg', 'qual_reg', 'cost_reg'],
					help='Metric to be plotted rn.')
parser.add_argument('--save-dir', type=str, help='The directory to save the plots in.')
args = parser.parse_args()

# Retrieve the metric to be plotted
my_metric = args.metric
# Retrieve y-label based on metric name
if my_metric == "reg":
	y_label = "Regret"
elif my_metric == "qual_reg":
	y_label = "Quality Regret"
elif my_metric == "cost_reg":
	y_label = "Cost Regret"
else:
	raise ValueError("Invalid metric name")

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

# Retrieve the number of files to be processed
sorted_files = sorted(pathlib.Path(log_folder).iterdir(), key=lambda x: x.name)
num_files = len(sorted_files)
# Init a numpy array to hold all the data
y_points = np.zeros((nalgos, num_files))

# Iterate over all the .csv files present in this folder
for file_idx, in_file in enumerate(sorted_files):
	# Extract out just file name sans extension
	in_name_raw = in_file.name.split('.')[0]
	# Ignore last 4 characters of log file name since they shall be _log
	in_name = in_name_raw[:-4]

	# Data Reading and Preprocessing
	# - - - - - - - - - - - -
	# Read in the log file as a pandas dataframe
	bandit_data = pd.read_csv(in_file, sep=",")
	# Infer the horizon as the largest entry in the 'horizon' column
	horizon_new = bandit_data["time-step"].max()
	# - - - - - - - - - - - -

	# Plotting
	# - - - - - - - - - - - -
	# If horizon being set for the first time, set it directly, else check if it is the same as the previous horizon
	if horizon == -1:
		horizon = horizon_new
	elif horizon != horizon_new:
		raise ValueError("Horizon mismatch in log files")

	for index, label in enumerate(selected_algos):
		algo_data = bandit_data[bandit_data["algo"] == label]
		y_point = algo_data.loc[algo_data["time-step"] == horizon][my_metric].mean()
		y_points[index, file_idx] = y_point

# Plot once all the data has been collected together by iterating over all the .csv files
plt.figure(figsize=(10, 10))


# Hard code set the x-ticks (These xticks are based on the line that equally spaces a reward in the instance
# generation bash script)
xticks = np.linspace(0.3, 0.6, 11)

for index in range(nalgos):
	plt.plot(xticks, y_points[index, :], linewidth=3, marker='o', markersize=20, linestyle='-', color=COLORS[index])

# Retrieve tha path for the directory to save the plots in
save_dir = args.save_dir

plt.legend(selected_algos, fontsize='large')  # Increase font size for legend
plt.xlabel("Quality of Cheaper Arm", fontweight="bold", fontsize=14)  # Increase font size for x-axis label
plt.ylabel(y_label, fontweight="bold", fontsize=14)  # Increase font size for y-axis label
# Set font size of ticks
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title("Policy Comparisons", fontweight="bold", fontsize=16)
# - - - - - - - - - - - -

# Save figure
plt.savefig(save_dir + "/{0}_{1}".format(in_name, my_metric) + ".png", bbox_inches="tight")
plt.close()
