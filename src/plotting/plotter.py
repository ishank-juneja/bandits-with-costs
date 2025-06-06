import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import argparse
import pathlib


# Command line input
parser = argparse.ArgumentParser()
parser.add_argument("--log-file", action="store", dest="file")
parser.add_argument('--algos', type=str, nargs='+',
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

# Data Reading and Preprocessing
# - - - - - - - - - - - -
# Read in the log file as a pandas dataframe
bandit_data = pd.read_csv(in_file, sep=",")
# Infer the horizon as the largest entry in the 'horizon' column
horizon = bandit_data["time-step"].max()
# Infer the time-step size of the simulation as the difference between the first two entries in the 'horizon' column
plot_step = bandit_data["time-step"].iloc[1] - bandit_data["time-step"].iloc[0]
# Convert 'nsamps' from string to NumPy array
bandit_data['nsamps'] = bandit_data['nsamps'].apply(lambda x: np.fromstring(x, dtype=int, sep=';'))
# - - - - - - - - - - - -

# Plotting
# - - - - - - - - - - - -
x_points = np.arange(0, horizon + 1, plot_step)
y_points = np.zeros_like(x_points)	# To hold the mean y-value
y_points_max = np.zeros_like(x_points)	# To hold the standard deviation of y-value

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
plot_handles = []	# List to store line object for the legend

for index, label in enumerate(selected_algos):
	# Extract out the data just for this algorithm out of the entire bandit_data
	algo_data = bandit_data[bandit_data["algo"] == label]

	# Identify the random seed that gave the maximum value of my_metric at the terminal time-step
	# Collect the terminal algorithm data into a variable
	terminal_data = algo_data[algo_data["time-step"] == horizon]

	# Get the index of the maximum entry in the my_metric column
	max_index = terminal_data[my_metric].idxmax()

	# Retrieve the corresponding 'rs' value
	max_seed = terminal_data.loc[max_index, 'rs']
	# Get the data-points for the maximum seed
	y_points_max = algo_data[(algo_data['rs'] == max_seed) & (algo_data['time-step'].isin(x_points))][my_metric].values
	# Compute the average regret at every plot_step interval time-step in a loop
	for i in range(len(y_points)):
		try:
			y_points[i] = algo_data.loc[algo_data["time-step"] == x_points[i]][my_metric].mean()
		except ValueError:
			print("Error: Likely data for some algorithm selected in --algos is missing. Exiting...")
			exit(-1)

	# Plot the mean line
	line, = plt.plot(x_points, y_points, color=COLORS[index], linewidth=3, label=label)
	plot_handles.append(line)

	# Plot the max line
	plt.plot(x_points, y_points_max, color=COLORS[index], linestyle='dashed', linewidth=1)

	# Calculate the lower and upper bounds for the 2\sigma interval
	# lower_bound = y_points - 2 * y_points_std_dev
	# upper_bound = y_points + 2 * y_points_std_dev

	# Fill the 2\sigma interval
	# plt.fill_between(x_points, lower_bound, upper_bound, color=COLORS[index], alpha=0.2)

# Retrieve tha path for the directory to save the plots in
save_dir = args.save_dir

plt.legend(handles=plot_handles, fontsize='large')  # Increase font size for legend
plt.xlabel("Steps $t$", fontweight="bold", fontsize=14)  # Increase font size for x-axis label
plt.ylabel(y_label, fontweight="bold", fontsize=14)  # Increase font size for y-axis label
# Set font size of ticks
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title("Policy Comparisons", fontweight="bold", fontsize=16)
# - - - - - - - - - - - -

# Save figure

# plt.show()

plt.savefig(save_dir + "/{0}_{1}".format(in_name, my_metric) + ".png", bbox_inches="tight")
plt.close()
