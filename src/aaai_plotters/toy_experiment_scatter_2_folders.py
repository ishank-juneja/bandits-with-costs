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
# The folder that contains the example where we are getting Linear Cost Regret
# This lin-cost-regret family has been set up such that there is no quality regret
parser.add_argument("--folder1", action="store", dest="folder1")
# The folder that contains the example where we are getting Linear Quality Regret
# This lin-qual-regret family has been set up such that there is no quality regret
parser.add_argument("--folder2", action="store", dest="folder2")
parser.add_argument('--algos', type=str, nargs='+', help='Algorithms for regret to be plotted')
parser.add_argument('--save-dir', type=str, help='The directory to save the plots in.', dest='save_dir')
args = parser.parse_args()

folder1 = args.folder1
folder2 = args.folder2
selected_algos = args.algos
save_dir = args.save_dir

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
sorted_files_folder1 = sorted(pathlib.Path(folder1).iterdir(), key=lambda x: x.name)
num_files1 = len(sorted_files_folder1)
assert num_files1 == 12, "Number of files in Linear Cost Regret experiment should be 12"
sorted_files_folder2 = sorted(pathlib.Path(folder2).iterdir(), key=lambda x: x.name)
num_files2 = len(sorted_files_folder2)
# assert num_files2 == 8, "Number of files in Linear Quality Regret experiment should be 8"


# Create a 2 x 1 figure. First we shall do all the things that go into plotting the
fig, axs = plt.subplots(2, 1, figsize=(8, 10))  # 2 rows, 1 column, figure size 8x10 inches
# Fix x-tick positions to hold the labels for the bandit instance
#  with the varying arm
# - - - - - - - - - - - - - -
# 12 equispaced points between 0 and 1 (inclusive)
plot1_xs = np.linspace(0, 1, num_files1)
# This below line needs to be hand-coded based on the plots
plot1_xs_markers = np.linspace(0.6, 0.93, 12) # The value of the varying first arm for the instance family
# Convert the NumPy array to a list of strings, formatted to 2 decimal places
plot1_xs_markers_str = [f'{x:.2f}' for x in plot1_xs_markers]
# 8 equi-spaced points between 0 and 1 (inclusive)
plot2_xs = np.linspace(0, 1, num_files2)  #
# This below line needs to be hand-coded based on the plots
plot2_xs_markers = np.linspace(0.01, 0.15, num_files2)  # The value of the varying second arm for the instance family
plot2_xs_markers_str = [f'{x:.2f}' for x in plot2_xs_markers]

axs[0].set_title('Toy Experiment 1')
axs[0].set_xlabel('Expected Return of Varying Arm')
axs[0].set_ylabel('Cost Regret')
axs[0].set_xticks(plot1_xs)
axs[0].set_xticklabels(plot1_xs_markers_str)  # Setting alphabetical x-tick labels

# Scatter plot on the second subplot
axs[1].set_title('Toy Experiment 2')
axs[1].set_xlabel('Expected Return of Varying Arm')
axs[1].set_ylabel('Quality Regret')
axs[1].set_xticks(plot2_xs)
axs[1].set_xticklabels(plot2_xs_markers_str)  # Setting alphabetical x-tick labels

# Part 1 START
# - - - - - - - - - -
# Linear Cost Regret Experiment Family
# Infer the random seed and ensure its consistency across all the log files
for file_idx, in_file in enumerate(sorted_files_folder1):
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

# Init an array to hold the cost files data
cost_regret_data = np.zeros((nalgos, num_files1, nseeds))
qual_regret_data = np.zeros((nalgos, num_files1, nseeds))
summed_regret_data = np.zeros((nalgos, num_files1, nseeds))

# - - - - - - - - - - - - - -
# Linear Cost Regret Experiment Family
# - - - - - - - - - - - - - -
# Plot the linear cost regret experiment family on the top plot
# Iterate over all the .csv files present in this folder
for file_idx, in_file in enumerate(sorted_files_folder1):
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
        cost_regret_data[index, file_idx, :] = algo_data.loc[algo_data["time-step"] == horizon]['cost_reg']
        qual_regret_data[index, file_idx, :] = algo_data.loc[algo_data["time-step"] == horizon]['qual_reg']

summed_regret_data = cost_regret_data + qual_regret_data

# Scatter plot on the first subplot
for idx in range(nalgos):
    for jdx, mark_x in enumerate(plot1_xs):
        axs[0].scatter([mark_x] * nseeds, summed_regret_data[idx, jdx], color=COLORS[idx], marker=marker_styles[idx], s=100, alpha=0.1)
        # Plot a single point per algorithm per x marker with the average:
        #  - Average over all seeds
        # axs[0].scatter(mark_x, np.mean(cost_regret_data[idx, jdx]), color='k', marker=marker_styles[idx], s=100)
# Create legend elements
legend_elements = [Line2D([0], [0], marker=marker_styles[i], color='w', label=selected_algos[i],
                          markerfacecolor=COLORS[i], markersize=10) for i in range(nalgos)]

# Add legend to the first subplot
axs[0].legend(handles=legend_elements, title="Algorithms", loc='upper right')
# Part 1 END

# Part 2 START
# - - - - - - - - - -
# Initialize nseeds with -1
nseeds = -1

# Linear Quality Regret Experiment Family
# Infer the random seed and ensure its consistency across all the log files
for file_idx, in_file in enumerate(sorted_files_folder2):
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

# Init an array to hold the quality files data
cost_regret_data = np.zeros((nalgos, num_files2, nseeds))
qual_regret_data = np.zeros((nalgos, num_files2, nseeds))
summed_regret_data = np.zeros((nalgos, num_files2, nseeds))

# - - - - - - - - - - - - - -
# Linear Quality Regret Experiment Family
# - - - - - - - - - - - - - -
# Plot the linear quality regret experiment family on the bottom plot
# Iterate over all the .csv files present in this folder
for file_idx, in_file in enumerate(sorted_files_folder2):
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
        cost_regret_data[index, file_idx, :] = algo_data.loc[algo_data["time-step"] == horizon]['cost_reg']
        qual_regret_data[index, file_idx, :] = algo_data.loc[algo_data["time-step"] == horizon]['qual_reg']

summed_regret_data = cost_regret_data + qual_regret_data

# Scatter plot on the second subplot
for idx in range(nalgos):
    for jdx, mark_x in enumerate(plot2_xs):
        axs[1].scatter([mark_x] * nseeds, summed_regret_data[idx, jdx], color=COLORS[idx], marker=marker_styles[idx], s=100, alpha=0.1)
        # Plot a single point per algorithm per x marker with the average:
        #  - Average over all seeds
        # axs[1].scatter(mark_x, np.mean(qual_regret_data[idx, jdx]), color='k', marker=marker_styles[idx], s=100)

# Create legend elements for the second plot
legend_elements = [Line2D([0], [0], marker=marker_styles[i], color='w', label=selected_algos[i],
                          markerfacecolor=COLORS[i], markersize=10) for i in range(nalgos)]

# Add legend to the second subplot
axs[1].legend(handles=legend_elements, title="Algorithms", loc='upper right')
# Part 2 END

# Adjust layout to prevent overlap
plt.tight_layout()

plt.show()

# Save figure
# plt.savefig(save_dir + "/toy/toy_experiment_added_together.pdf", bbox_inches="tight")
# plt.close()
