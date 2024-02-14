import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import argparse
import pathlib


# Command line input
parser = argparse.ArgumentParser()
parser.add_argument("--log-file", action="store", dest="file")
parser.add_argument('--algorithms', type=str, nargs='+',
                    help='Algorithms for regret to be plotted')
parser.add_argument('--metric', type=str, choices=['reg', 'qual_reg', 'cost_reg'],
					help='Metric to be plotted rn.')
parser.add_argument('--save-dir', type=str, help='The directory to save the plots in.')
args = parser.parse_args()

in_file = args.file
# Extract out just file name sans extension
in_name = in_file.split('/')[-1].split('.')[0]
# Ignore last 4 characters of log file name since they shall be _log
in_name = in_name[:-4]
# Get list of the algorithms to be plotted
selected_algos = args.algorithms
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

# Color Management
# - - - - - - - - - - - -
# Read in the log file as a pandas dataframe
bandit_data = pd.read_csv(in_file, sep=",")
# Infer the horizon as the largest entry in the 'horizon' column
horizon = bandit_data["horizon"].max()
# Infer the time-step size of the simulation as the difference between the first two entries in the 'horizon' column
plot_step = bandit_data["horizon"].iloc[1] - bandit_data["horizon"].iloc[0]
# Convert 'nsamps' from string to NumPy array
bandit_data['nsamps'] = bandit_data['nsamps'].apply(lambda x: np.fromstring(x, dtype=int, sep=';'))
# - - - - - - - - - - - -

# Plotting
# - - - - - - - - - - - -
x_points = np.arange(0, horizon + 1, plot_step)
y_points = np.zeros_like(x_points)

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

# Plot the chosen data
plt.figure(figsize=(10, 10))
for index, label in enumerate(selected_algos):
	algo_data = bandit_data[bandit_data["algo"] == label]
	for i in range(len(y_points)):
		y_points[i] = algo_data.loc[algo_data["horizon"] == x_points[i]][my_metric].mean()
	plt.plot(x_points, y_points, color=COLORS[index], linewidth=3)
# Retrieve tha path for the directory to save the plots in
save_dir = args.save_dir

plt.legend(selected_algos, fontsize='large')  # Increase font size for legend
plt.xlabel("Steps $t$", fontweight="bold", fontsize=14)  # Increase font size for x-axis label
plt.ylabel(y_label, fontweight="bold", fontsize=14)  # Increase font size for y-axis label
# Set font size of ticks
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title("Policy Comparisons", fontweight="bold", fontsize=16)
# - - - - - - - - - - - -

# Save figure
plt.savefig(save_dir + "/{0}_{1}".format(in_name, my_metric) + ".png", bbox_inches="tight")
plt.close()














