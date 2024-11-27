import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import argparse
from matplotlib.ticker import FuncFormatter
# For Type 3 free fonts in the figures
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
import matplotlib.ticker as mticker

# Command line input
parser = argparse.ArgumentParser()
parser.add_argument("--log-file", action="store", dest="file")
parser.add_argument('--save-dir', type=str, help='The directory to save the plots in.')
args = parser.parse_args()

selected_algos = ['pe', 'ucb-cs']

in_file = args.file
in_name = in_file.split('/')[-1].split('.')[0][:-4]

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
marker_styles = ['o', 's', '^', '*']

bandit_data = pd.read_csv(in_file, sep=",")
horizon = bandit_data["time-step"].max()
plot_step = bandit_data["time-step"].iloc[1] - bandit_data["time-step"].iloc[0]
bandit_data['nsamps'] = bandit_data['nsamps'].apply(lambda x: np.fromstring(x, dtype=int, sep=';'))

# Creating the figure and subplots
fig, ax = plt.subplots(figsize=(8, 5))

# Set the width of the spines for each subplot
spine_width = 1  # Thickness of the border
for spine in ax.spines.values():
    spine.set_linewidth(spine_width)
# Increase tick label font size
ax.tick_params(axis='both', labelsize=14)  # Set font size

def thousands_formatter(x, pos):
    return '%1.0fK' % (x * 1e-3)


def millions_formatter(x, pos):
    return '%1.0fM' % (x * 1e-6)

# Plot each trend line one at a time with deliberate choice of color, marker
# - - - - - - - - - - - - - - -

# Same x-points used for all the plots
x_points = np.arange(0, horizon + 1, plot_step)

# Average cost regret
y_points = np.zeros_like(x_points, dtype=float)  # Reset y_points
for idx, algo_name in enumerate(selected_algos):
    # Filter out the data corresponding to just this algorithm
    algo_data = bandit_data[bandit_data["algo"] == algo_name]
    # Populate y_points in a loop
    for jdx in range(len(y_points)):
        cost_reg_jdx_data = algo_data.loc[algo_data["time-step"] == x_points[jdx]]['cost_reg'].mean()
        qual_reg_jdx_data = algo_data.loc[algo_data["time-step"] == x_points[jdx]]['qual_reg'].mean()
        y_points[jdx] = cost_reg_jdx_data + qual_reg_jdx_data
    ax.plot(x_points, y_points, color=COLORS[idx], linewidth=3, label=algo_name,
                   marker=marker_styles[idx], markersize=10, markevery=500)
ax.set_title(r'Average Regret', fontweight="bold", fontsize=18)
# - - - - - - - - - - - - - - -

# Ensuring axs[0] and axs[1] share the same y-axis
y0_min, y0_max = ax.get_ylim()
ax.set_ylim(y0_min, y0_max)

# Add shared x-axis label
ax.set_xlabel('Time $(t)$', fontsize=16, fontweight='bold')

# Create an instance of FuncFormatter using your custom function
formatter_1K = FuncFormatter(thousands_formatter)
formatter_1M = FuncFormatter(millions_formatter)

# Apply this formatter_1K to the y-axis of each subplot
# ax.yaxis.set_major_formatter(formatter_1K)
# ax.set_xticklabels(ax.get_xticks(), fontsize=16, fontweight='bold') # Setting alphabetical x-tick labels  # Setting alphabetical x-tick labels
ax.set_xticklabels(ax.get_xticks(), fontsize=16) # Setting alphabetical x-tick labels  # Setting alphabetical x-tick labels
ax.xaxis.set_major_formatter(formatter_1M)
# ax.set_yticklabels(ax.get_yticks(), fontsize=16, fontweight='bold') # Setting alphabetical x-tick labels  # Setting alphabetical x-tick labels
ax.set_yticklabels(ax.get_yticks(), fontsize=16) # Setting alphabetical x-tick labels  # Setting alphabetical x-tick labels
ax.yaxis.set_major_formatter(formatter_1K)

# Set the shared Common Y axis Label
ax.set_ylabel('Cost Regret + Quality Regret', fontsize=16, labelpad=10, fontweight='bold')

# Once all plots are fully set up
# No need to redraw the canvas here, as all settings are finalized
ax.grid(which='both', axis='both', color='k', alpha=0.2)

# Place the legend on the top left sub-plot, that is axs[0, 0]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper left', fontsize=16, framealpha=1.0,
          handlelength=3, columnspacing=0.5)

# plt.show()

plt.savefig(f"{args.save_dir}/figure1.pdf", bbox_inches="tight")
plt.close()
