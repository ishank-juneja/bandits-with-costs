import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import argparse
import pathlib
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

# Command line input
parser = argparse.ArgumentParser()
parser.add_argument("--log-file", action="store", dest="file")
parser.add_argument('--save-dir', type=str, help='The directory to save the plots in.')
args = parser.parse_args()

selected_algos = ['cs-pe', 'cs-ucb', 'cs-ts', 'cs-etc']

# Map algo names used in the log files to algo names used in the writing
custom_algo_names = {
    'cs-pe': 'pe-cs',
    'cs-ucb': 'ucb-cs',
    'cs-ts': 'ts-cs',
    'cs-etc': 'etc-cs'
}

in_file = args.file
in_name = in_file.split('/')[-1].split('.')[0][:-4]

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
marker_styles = ['o', 's', '^', '*']

bandit_data = pd.read_csv(in_file, sep=",")
horizon = bandit_data["time-step"].max()
plot_step = bandit_data["time-step"].iloc[1] - bandit_data["time-step"].iloc[0]
bandit_data['nsamps'] = bandit_data['nsamps'].apply(lambda x: np.fromstring(x, dtype=int, sep=';'))

# Creating the figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 14))  # 2x2 grid of axes
plt.subplots_adjust(wspace=0, hspace=0, right=0.85, left=0.1)  # Set wspace to 0 to remove horizontal space between plots

# Set the width of the spines for each subplot
spine_width = 3  # Thickness of the border

for ax in axs.flat:
    # Set the spine width
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    # Increase tick label font size
    ax.tick_params(axis='both', labelsize=14)  # Set font size

# Remove y-axis ticks and labels for the right column subplots twice
for col in [1]:
    for row in [0, 1]:
        axs[row, col].tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

# Remove y-axis ticks and labels for the right column subplots
axs[0, 1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
axs[1, 1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

# Remove y-axis ticks and labels for the right column subplots
axs[0, 1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
axs[1, 1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

metrics = ["cost_reg", "qual_reg"]
labels = ["Cost Regret", "Quality Regret"]


def thousands_formatter(x, pos):
    return '%1.0fK' % (x * 1e-3)

# Plot each trend line one at a time with deliberate choice of color, marker
# - - - - - - - - - - - - - - -

# Same x-points used for all the plots
x_points = np.arange(0, horizon + 1, plot_step)

# Average cost regret
y_points = np.zeros_like(x_points)
for idx, algo_name in enumerate(selected_algos):
    # Filter out the data corresponding to just this algorithm
    algo_data = bandit_data[bandit_data["algo"] == algo_name]
    # Populate y_points in a loop
    for jdx in range(len(y_points)):
        y_points[jdx] = algo_data.loc[algo_data["time-step"] == x_points[jdx]]['cost_reg'].mean()
    axs[0, 0].plot(x_points, y_points, color=COLORS[idx], linewidth=3, label=custom_algo_names[algo_name],
                   marker=marker_styles[idx], markersize=10, markevery=500)
    axs[0, 0].set_title('Cost Regret', fontweight="bold", fontsize=14)
# - - - - - - - - - - - - - - -

# Worst cost regret
y_points = np.zeros_like(x_points) # Reset y_points
for idx, algo_name in enumerate(selected_algos):
    # Filter out the data corresponding to just this algorithm
    algo_data = bandit_data[bandit_data["algo"] == algo_name]
    # Populate y_points with the worst case performance
    max_index = algo_data[algo_data["time-step"] == horizon]['cost_reg'].idxmax()
    max_seed = algo_data.loc[max_index, 'rs']
    y_points = algo_data[(algo_data['rs'] == max_seed) & (algo_data['time-step'].isin(x_points))]['cost_reg'].values
    axs[1, 0].plot(x_points, y_points, color=COLORS[idx], linewidth=3, label=custom_algo_names[algo_name],
                   marker=marker_styles[idx], markersize=10, markevery=500)
# - - - - - - - - - - - - - - -

# Average quality regret
y_points = np.zeros_like(x_points)
for idx, algo_name in enumerate(selected_algos):
    # Filter out the data corresponding to just this algorithm
    algo_data = bandit_data[bandit_data["algo"] == algo_name]
    # Populate y_points in a loop
    for jdx in range(len(y_points)):
        y_points[jdx] = algo_data.loc[algo_data["time-step"] == x_points[jdx]]['qual_reg'].mean()
    axs[0, 1].plot(x_points, y_points, color=COLORS[idx], linewidth=3, label=custom_algo_names[algo_name],
                   marker=marker_styles[idx], markersize=10, markevery=500)
    axs[0, 1].set_title('Quality Regret', fontweight="bold", fontsize=14)
# - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - -
# Worst quality regret
y_points = np.zeros_like(x_points)
for idx, algo_name in enumerate(selected_algos):
    # Filter out the data corresponding to just this algorithm
    algo_data = bandit_data[bandit_data["algo"] == algo_name]
    # Populate y_points with the worst case performance
    max_index = algo_data[algo_data["time-step"] == horizon]['qual_reg'].idxmax()
    max_seed = algo_data.loc[max_index, 'rs']
    y_points = algo_data[(algo_data['rs'] == max_seed) & (algo_data['time-step'].isin(x_points))]['qual_reg'].values
    axs[1, 1].plot(x_points, y_points, color=COLORS[idx], linewidth=3, label=custom_algo_names[algo_name],
                   marker=marker_styles[idx], markersize=10, markevery=500)
# - - - - - - - - - - - - - - -

# Ensuring axs[0, 0] and axs[0, 1] share the same y-axis
y0_min, y0_max = axs[0, 0].get_ylim()
y1_min, y1_max = axs[0, 1].get_ylim()
shared_y0_max = max(y0_max, y1_max)
shared_y0_min = min(y0_min, y1_min)
axs[0, 0].set_ylim(shared_y0_min, shared_y0_max)
axs[0, 1].set_ylim(shared_y0_min, shared_y0_max)

# Ensuring axs[1, 0] and axs[1, 1] share the same y-axis
y2_min, y2_max = axs[1, 0].get_ylim()
y3_min, y3_max = axs[1, 1].get_ylim()
shared_y1_max = 1.08 * max(y2_max, y3_max) # Added some % manually
# for white space between top and bottom row of plots markers
shared_y1_min = min(y2_min, y3_min)
axs[1, 0].set_ylim(shared_y1_min, shared_y1_max)
axs[1, 1].set_ylim(shared_y1_min, shared_y1_max)

# Add shared x-axis label
fig.text(0.5, 0.04, 'Time Step $t$', ha='center', va='center', fontsize=14, fontweight='bold')

# Create an instance of FuncFormatter using your custom function
formatter = FuncFormatter(thousands_formatter)

# Apply this formatter to the y-axis of each subplot
axs[0, 0].yaxis.set_major_formatter(formatter)
axs[0, 1].yaxis.set_major_formatter(formatter)
axs[1, 0].yaxis.set_major_formatter(formatter)
axs[1, 1].yaxis.set_major_formatter(formatter)

# Increase title font size and remove bold style
for ax, label in zip(axs.flat, labels):
    ax.set_title(label, fontsize=16, fontweight='normal')

# Once all plots are fully set up
# Add grid to each subplot aligned with the actual ticks
for ax in axs.flat:
    # No need to redraw the canvas here, as all settings are finalized
    ax.grid(which='both', axis='both', color='k', alpha=0.2)

# Update legend to be 2x2
handles, labels = axs[0, 0].get_legend_handles_labels()
axs[0, 0].legend(handles, labels, loc='upper left', ncol=2, fontsize=12)

plt.savefig(f"{args.save_dir}/movie_lens_experiment.pdf", bbox_inches="tight")
plt.close()
