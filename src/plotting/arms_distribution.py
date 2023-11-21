from gif_maker import GIFMaker
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter
import numpy as np
import argparse
import pathlib


# Command line input
parser = argparse.ArgumentParser()
parser.add_argument("-idx", action="store", dest="file")
parser.add_argument("-STEP", action="store", dest="STEP", type=int)
parser.add_argument("-horizon", action="store", dest="horizon", type=float)
args = parser.parse_args()
in_file = args.file
# Extract out just file name
in_name = in_file.split('/')[-1].split('.')[0]
# Ignore last 4 characters -out
in_name = in_name[:-4]
# Create folder to save results if it doesn't already exist
pathlib.Path('results/' + in_name).mkdir(parents=False, exist_ok=True)
x_points = np.arange(args.STEP, int(args.horizon) + 1, args.STEP)
y_points = np.zeros_like(x_points)
# These are all the possible labels, of these at most 8 can be supported with below colors
# LABELS = ['ucb', 'ts', 'qucb', 'qts', 'cucb', 'cts', 'u-cucb', 'new', 'cts-old', 'cucb-old']
# selected_algos must be a subset of algos in simulate_policies.py
# All the algorithms selected_algos must have already been simulated in simulate_policies.py
selected_algos = ['ucb', 'ucb-cs']
# Number of distinct algorithms used
nalgos = len(selected_algos)
# Number of colors should be at least as many as number of LABELS
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
# Choose as many colors as there are algorithms selected_algos
if len(COLORS) < len(selected_algos):
	print("Error: Not enough colors present for plotting\nAdd colors or reduce number of algorithms")
	exit(-1)
else:
	COLORS = COLORS[:len(selected_algos)]

bandit_data = pd.read_csv(in_file, sep=",", header=None)
bandit_data.columns = ["algo", "rs", "horizon", "qual_reg", "cost_reg", "nsamps"]

# First, determine narms and N-plotted-steps
narms = len(bandit_data['nsamps'].iloc[0].split(';'))  # Assuming all rows have the same number of arms
N_plotted_steps = len(x_points)

# init array to hold nsamps data for the time-steps we plot for
nsamps_array = np.zeros((len(selected_algos), N_plotted_steps, narms))

# Populate the array with data from the dataframe
for idx, t in enumerate(x_points):
    for algo_index, label in enumerate(selected_algos):
        cur_data = bandit_data.loc[(bandit_data["algo"] == label) & (bandit_data["horizon"] == t)]
        if not cur_data.empty:
            nsamps_values = np.fromstring(cur_data.iloc[0]['nsamps'], dtype=int, sep=';')
            nsamps_array[algo_index, idx, :] = nsamps_values

# Create temporary directory for frames
tmp_frames_dir = '../../results/gifs/tmp'
pathlib.Path(tmp_frames_dir).mkdir(parents=True, exist_ok=True)

# Find the maximum value across all data points for setting y-axis limits
max_pulls = np.max(nsamps_array)
if max_pulls <= 0:
    max_pulls = 1  # Ensure max_pulls is greater than 0
# Determine the width of each bar
bar_width = 1 / (nalgos + 1)  # Adding 1 for some spacing between groups of bars

# Loop through each STEP and create a bar chart for each
for idx, t in enumerate(x_points):
    plt.figure(figsize=(10, 10))
    for algo_index, label in enumerate(selected_algos):
        # Calculate the x-position for each bar
        x_positions = np.arange(narms) + (algo_index * bar_width)

        plt.bar(x_positions, nsamps_array[algo_index, idx, :], width=bar_width, color=COLORS[algo_index], alpha=0.5,
                label=label)

    plt.xlabel("Arms", fontweight="bold")
    plt.ylabel("Number of Pulls", fontweight="bold")
    plt.title(f"Distribution of Arm Pulls at Time {t}", fontweight="bold")
    plt.legend()
    plt.xticks(np.arange(narms) + bar_width / 2, np.arange(narms) + 1)

    plt.yscale('log')  # Change to log scale
    plt.ylim(1, max_pulls * 10)  # Set consistent Y-axis limits for all plots

    # Save frame as PNG in tmp_frames_dir
    frame_filename = os.path.join(tmp_frames_dir, "file{0:05d}.png".format(idx + 1))
    plt.savefig(frame_filename, bbox_inches='tight')
    plt.close()

# Use GIFMaker to create GIF from frames
gif_maker = GIFMaker(delay=40)
gif_path = f'results/{in_name}/{in_name}_nsamps_distribution.gif'
gif_maker.make_gif(gif_path, tmp_frames_dir)

gif_maker.cleanup_frames(tmp_frames_dir)
