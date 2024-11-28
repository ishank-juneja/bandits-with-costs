# Plotter for the experiment where we sweep the return of one arm
#  while keeping the other arms in the instance fixed
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import argparse
import pathlib
from matplotlib.lines import Line2D
# For Type 3 free fonts in the figures
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Command line input
parser = argparse.ArgumentParser()
# The folder that contains the example where we are getting Linear Cost Regret
# This lin-cost-regret family has been set up such that there is no quality regret
parser.add_argument("--folder", action="store", dest="folder")
# algos should be a subset of: pe ucb-cs asymmetric-pe
parser.add_argument('--algos', type=str, nargs='+', help='Algorithms for regret to be plotted')
parser.add_argument('--save-dir', type=str, help='The directory to save the plots in.', dest='save_dir')
args = parser.parse_args()

folder = args.folder
selected_algos = args.algos
save_dir = args.save_dir

# Use difference scatter plot marker style for every algorithm
marker_styles = ['o', 's', '^', '*']
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

nalgos = len(selected_algos)

# Initialize nseeds with -1
nseeds = -1

# Retrieve the number of files to be processed
sorted_files_folder1 = sorted(pathlib.Path(folder).iterdir(), key=lambda x: x.name)
num_files1 = len(sorted_files_folder1)
assert num_files1 == 7, "Hard Coded 4 different values of reference arm ell"


# Create a 2 x 1 figure. First we shall do all the things that go into plotting the
fig, axs = plt.subplots(1, 1, figsize=(10, 6))
# Fix x-tick positions to hold the labels for the bandit instance
#  with the varying arm
# - - - - - - - - - - - - - -
# 12 equispaced points between 0 and 1 (inclusive)
plot1_xs = np.linspace(0, 1, num_files1)
# This below line needs to be hand-coded based on the plots
plot1_xs_markers = np.array(range(1, 20, 3)) # The value of the varying first arm for the instance family
# Convert the NumPy array to a list of strings, formatted to 2 decimal places
plot1_xs_markers_str = [f'{x:.2f}' for x in plot1_xs_markers]

axs.set_title('Hyper-parameter for Known Ell Experiment', fontsize=18, fontweight='bold', pad=10)
axs.set_xlabel(r'Arm $\ell$ index', fontsize=16, labelpad=10, fontweight='bold')
axs.set_ylabel('Cost Regret + Quality Regret', fontsize=16, labelpad=10, fontweight='bold')
axs.set_xticks(plot1_xs)
# axs.set_xticklabels(plot1_xs_markers_str, fontsize=16, fontweight='bold') # Setting alphabetical x-tick labels  # Setting alphabetical x-tick labels
axs.set_xticklabels(plot1_xs_markers_str, fontsize=16) # Setting alphabetical x-tick labels  # Setting alphabetical x-tick labels
# axs.set_yticklabels(axs.get_yticks(), fontsize=16, fontweight='bold')
axs.set_yticklabels(axs.get_yticks(), fontsize=16)

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
for idx, algo_name in enumerate(selected_algos):
    for jdx, mark_x in enumerate(plot1_xs):
        axs.scatter([mark_x] * nseeds, summed_regret_data[idx, jdx], color=COLORS[idx], marker=marker_styles[idx],
                    s=100, alpha=1.0, label=algo_name)
        # Plot a single point per algorithm per x marker with the average:
        #  - Average over all seeds
        # axs.scatter(mark_x, np.mean(cost_regret_data[idx, jdx]), color='k', marker=marker_styles[idx], s=100)
# Apply logarithmic scale to the y-axis
axs.set_yscale('log')


# Add legend to the first subplot
legend_elements = [Line2D([0], [0], marker=marker_styles[i], color='w',
                          label=selected_algos[i], markerfacecolor=COLORS[i], markersize=18)
                   for i in range(nalgos)]
axs.legend(handles=legend_elements, loc='lower left', ncol=2, fontsize=16, columnspacing=0.2)

# Adjust layout to prevent overlap
plt.tight_layout()

plt.show()

# Save figure
# plt.savefig(save_dir + "/ml/ml_alpha_swept.pdf", bbox_inches="tight")
# plt.close()
