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

# Map algo names used in the log files to algo names used in the code
custom_algo_names = {
    'cs-pe': 'pe-cs',
    'cs-ucb': 'ucb-cs',
    'cs-ts': 'ts-cs',
    'cs-etc': 'etc-cs'
}

in_file = args.file
in_name = in_file.split('/')[-1].split('.')[0][:-4]
nalgos = len(selected_algos)

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
if len(COLORS) < nalgos:
    print("Error: Not enough colors present for plotting\nAdd colors or reduce number of algorithms")
    exit(-1)
else:
    COLORS = COLORS[:nalgos]

bandit_data = pd.read_csv(in_file, sep=",")
horizon = bandit_data["time-step"].max()
plot_step = bandit_data["time-step"].iloc[1] - bandit_data["time-step"].iloc[0]
bandit_data['nsamps'] = bandit_data['nsamps'].apply(lambda x: np.fromstring(x, dtype=int, sep=';'))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
plt.subplots_adjust(wspace=0, right=0.85, left=0.1)  # Set wspace to 0 to remove horizontal space between plots

metrics = ["cost_reg", "qual_reg"]
labels = ["Cost Regret", "Quality Regret"]


def thousands_formatter(x, pos):
    return '%1.0f' % (x * 1e-3)


for ax, metric, label in zip([ax1, ax2], metrics, labels):
    ax.set_title(label, fontweight="bold", fontsize=14)  # Setting the title instead of y-axis label
    x_points = np.arange(0, horizon + 1, plot_step)
    y_points = np.zeros_like(x_points)

    for index, algo_label in enumerate(selected_algos):
        algo_data = bandit_data[bandit_data["algo"] == algo_label]
        max_index = algo_data[algo_data["time-step"] == horizon][metric].idxmax()
        max_seed = algo_data.loc[max_index, 'rs']
        y_points_max = algo_data[(algo_data['rs'] == max_seed) & (algo_data['time-step'].isin(x_points))][metric].values

        for i in range(len(y_points)):
            y_points[i] = algo_data.loc[algo_data["time-step"] == x_points[i]][metric].mean()

        ax.plot(x_points, y_points, color=COLORS[index], linewidth=3, label=algo_label if ax is ax1 else "_nolegend_")
        ax.plot(x_points, y_points_max, color=COLORS[index], linestyle='dashed', linewidth=1, label="_nolegend_")

    ax.set_xlabel("Steps $t$", fontweight="bold", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label size
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))  # Format y-axis ticks

# Get existing handles and labels from the first plot
handles, labels = ax1.get_legend_handles_labels()
labels = [custom_algo_names[label] for label in labels]

# Append custom line styles to legend
handles.append(Line2D([0], [0], color='black', lw=2, linestyle='-', label='Average (solid)'))
handles.append(Line2D([0], [0], color='black', lw=2, linestyle='--', label='Best/Worst (dashed)'))

# Add updated legend to the first plot
ax1.legend(handles=handles, labels=labels + ['Average (solid)', 'Best/Worst (dashed)'], fontsize=14, loc='upper left')

plt.savefig(f"{args.save_dir}/movie_lens_experiment.pdf", bbox_inches="tight")
plt.close()
