import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import argparse
import pathlib
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
# For Type 3 free fonts in the figures
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
import matplotlib.ticker as mticker

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
fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 2x2 grid of axes
# Set wspace to 0 to remove horizontal space between plots
plt.subplots_adjust(wspace=0, hspace=0, right=0.98, left=0.05, top=0.95, bottom=0.1)

# Set the width of the spines for each subplot
spine_width = 1  # Thickness of the border

for ax in axs.flat:
    # Set the spine width
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    # Increase tick label font size
    ax.tick_params(axis='both', labelsize=22)  # Set font size

# Remove y-axis ticks and labels for the right column subplots twice
axs[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)


def thousands_formatter(x, pos):
    return '%1.0fK' % (x * 1e-3)


def millions_formatter(x, pos):
    return '%1.0fM' % (x * 1e-6)

# Plot each trend line one at a time with deliberate choice of color, marker
# - - - - - - - - - - - - - - -

# Same x-points used for all the plots
x_points = np.arange(0, horizon + 1, plot_step)

# Average cost regret
y_points = np.zeros_like(x_points, dtype=float)
for idx, algo_name in enumerate(selected_algos):
    # Filter out the data corresponding to just this algorithm
    algo_data = bandit_data[bandit_data["algo"] == algo_name]
    # Populate y_points in a loop
    for jdx in range(len(y_points)):
        cost_reg_jdx_data = algo_data.loc[algo_data["time-step"] == x_points[jdx]]['cost_reg'].mean()
        qual_reg_jdx_data = algo_data.loc[algo_data["time-step"] == x_points[jdx]]['qual_reg'].mean()
        y_points[jdx] = cost_reg_jdx_data + qual_reg_jdx_data
    axs[0].plot(x_points, y_points, color=COLORS[idx], linewidth=3, label=custom_algo_names[algo_name],
                   marker=marker_styles[idx], markersize=10, markevery=500)
    axs[0].set_title(r'Average Regret', fontweight="bold", fontsize=24, pad=10)
# - - - - - - - - - - - - - - -

# Worst cost regret
y_points = np.zeros_like(x_points) # Reset y_points
for idx, algo_name in enumerate(selected_algos):
    # Filter out the data corresponding to just this algorithm
    algo_data = bandit_data[bandit_data["algo"] == algo_name]
    # Populate y_points with the worst case performance
    max_index = algo_data[algo_data["time-step"] == horizon]['cost_reg'].idxmax()
    max_seed = algo_data.loc[max_index, 'rs']
    y_points_cost = algo_data[(algo_data['rs'] == max_seed) & (algo_data['time-step'].isin(x_points))]['cost_reg'].values
    y_points_qual = algo_data[(algo_data['rs'] == max_seed) & (algo_data['time-step'].isin(x_points))]['qual_reg'].values
    axs[1].plot(x_points, y_points_cost + y_points_qual, color=COLORS[idx], linewidth=3,
                label=custom_algo_names[algo_name], marker=marker_styles[idx], markersize=10, markevery=500)
    axs[1].set_title(r'Worst Regret', fontweight="bold", fontsize=24, pad=10)
# - - - - - - - - - - - - - - -

# Ensuring axs[0] and axs[1] share the same y-axis
y0_min, y0_max = axs[0].get_ylim()
y1_min, y1_max = axs[1].get_ylim()
shared_y0_max = max(y0_max, y1_max)
shared_y0_min = min(y0_min, y1_min)
axs[0].set_ylim(shared_y0_min, shared_y0_max)
axs[1].set_ylim(shared_y0_min, shared_y0_max)

# Add shared x-axis label
fig.text(0.5, 0.0, 'Time $(t)$', ha='center', va='center', fontsize=22, fontweight='bold')

# Create an instance of FuncFormatter using your custom function
formatter_1K = FuncFormatter(thousands_formatter)
formatter_1M = FuncFormatter(millions_formatter)

# Apply this formatter_1K to the y-axis of each subplot

axs[0].set_xticklabels(axs[0].get_xticks(), fontsize=22) # Setting alphabetical x-tick labels  # Setting alphabetical x-tick labels
axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=22) # Setting alphabetical x-tick labels  # Setting alphabetical x-tick labels
axs[1].set_xticklabels(axs[1].get_xticks(), fontsize=22) # Setting alphabetical x-tick labels  # Setting alphabetical x-tick labels
axs[1].set_yticklabels(axs[1].get_yticks(), fontsize=22) # Setting alphabetical x-tick labels  # Setting alphabetical x-tick labels
axs[0].xaxis.set_major_formatter(formatter_1M)
axs[0].yaxis.set_major_formatter(formatter_1K)
axs[1].xaxis.set_major_formatter(formatter_1M)
axs[1].yaxis.set_major_formatter(formatter_1K)

# Set the shared Common Y axis Label
axs[0].set_ylabel('Cost Regret + Quality Regret', fontsize=24, fontweight='bold')

# Once all plots are fully set up
# Add grid to each subplot aligned with the actual ticks
for ax in axs.flat:
    # No need to redraw the canvas here, as all settings are finalized
    ax.grid(which='both', axis='both', color='k', alpha=0.2)

# Place the legend on the top left sub-plot, that is axs[0, 0]
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles, labels, loc='upper left', ncol=2, fontsize=22,
                 framealpha=1.0, handlelength=3, columnspacing=0.5)

# plt.show()
plt.savefig(f"{args.save_dir}/figure2.pdf", bbox_inches="tight")
plt.close()
