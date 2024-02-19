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
parser.add_argument("--log-file", action="store", dest="file")
parser.add_argument('--algos', type=str, nargs='+',
                    help='Algorithms for which arm distributions are to be plotted')
parser.add_argument('--save-dir', type=str, help='The directory to save the GIF and tmp frames in.')
args = parser.parse_args()

in_file = args.file
# Extract out just file name
in_name = in_file.split('/')[-1].split('.')[0]
# Ignore last 4 characters -out
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
# - - - - - - - - - - -
# Read in the log file as a pandas dataframe
bandit_data = pd.read_csv(in_file, sep=",")
# Infer the horizon as the largest entry in the 'horizon' column
horizon = bandit_data["time-step"].max()
# Convert the pandas column for time-step to a NumPy array
full_t_points_redundant = bandit_data["time-step"].to_numpy()
# Find the time interval step-size used in the log file
plot_step = full_t_points_redundant[1] - full_t_points_redundant[0]
# Find the index of the first time a full horizon length of steps is reached
first_full_horizon_index = horizon // plot_step
full_t_points = full_t_points_redundant[:first_full_horizon_index + 1]

# Desired number of data points
total_desired_points = 80
first_segment_ratio = 0.1
second_segment_ratio = 0.9
first_segment_desired_points = 40
second_segment_desired_points = 40

# Calculate the number of points in the first 10%
first_segment_length = int(len(full_t_points) * first_segment_ratio)

# Calculate step size for each segment to evenly pick data points
first_segment_step_size = max(1, first_segment_length // first_segment_desired_points)
second_segment_step_size = max(1, (len(full_t_points) - first_segment_length) // second_segment_desired_points)

# Sub-sample the first 10% of the array
first_segment_points = full_t_points[:first_segment_length][::first_segment_step_size]
# Sub-sample the remaining 90% of the array
second_segment_points = full_t_points[first_segment_length:][::second_segment_step_size]

# Combine the two subsampled arrays
t_points = np.concatenate((first_segment_points, second_segment_points))

# Adjustments if the concatenated array has more points than desired due to rounding
if len(t_points) > total_desired_points:
    # Option to trim excess, or you could intelligently select points to remove
    t_points = t_points[:total_desired_points]
# t_points now contains 20 points from the first 10% and 60 points from the remaining 90%

# Convert 'nsamps' from string to NumPy array
bandit_data['nsamps'] = bandit_data['nsamps'].apply(lambda x: np.fromstring(x, dtype=int, sep=';'))
# - - - - - - - - - - - -

# Assemble the arm pull data into a np array for plotting frames
# - - - - - - - - - - - -
narms = len(bandit_data['nsamps'].iloc[0])  # Assuming all rows have the same number of arms
N_plotted_steps = len(t_points)
nsamps_array = np.zeros((len(selected_algos), N_plotted_steps, narms))
# Populate the array with data from the dataframe
for idx, t in enumerate(t_points):
    for algo_index, label in enumerate(selected_algos):
        cur_data = bandit_data.loc[(bandit_data["algo"] == label) & (bandit_data["time-step"] == t)]
        if not cur_data.empty:
            nsamps_array[algo_index, idx, :] = cur_data.iloc[0]['nsamps']
# - - - - - - - - - - - -

# Plot temporary frames for GIF
# - - - - - - - - - - - -
# Create temporary directory for frames
save_dir = args.save_dir
tmp_frames_dir = os.path.join(save_dir, "tmp")
pathlib.Path(tmp_frames_dir).mkdir(parents=True, exist_ok=True)

# Find the maximum value across all data points for setting y-axis limits
max_pulls = np.max(nsamps_array)
if max_pulls <= 0:
    max_pulls = 1  # Ensure max_pulls is greater than 0
# Determine the width of each bar
bar_width = 1 / (nalgos + 1)  # Adding 1 for some spacing between groups of bars

# Loop through each STEP and create a bar chart for each
for idx, t in enumerate(t_points):
    plt.figure(figsize=(10, 10))
    for algo_index, label in enumerate(selected_algos):
        # Calculate the x-position for each bar
        x_positions = np.arange(narms) + (algo_index * bar_width)
        # Retrieve the data-point to be plotted in the current iteration
        arm_pull_data_pt = nsamps_array[algo_index, idx, :]
        # If any entry is 0, set it to 1 to avoid log(0) error
        arm_pull_data_pt[arm_pull_data_pt == 0] = 1
        plt.bar(x_positions, arm_pull_data_pt, width=bar_width, color=COLORS[algo_index],
                alpha=0.5, label=label)

    plt.xlabel("Arms", fontweight="bold", fontsize=14)  # Increase font size for x-axis label
    plt.ylabel("Number of Pulls", fontweight="bold", fontsize=14)  # Increase font size for y-axis label
    plt.title(f"Distribution of Arm Pulls at Time {t}", fontweight="bold")
    plt.legend(fontsize='large')  # Increase font size for legend
    plt.xticks(np.arange(narms) + bar_width / 2, np.arange(narms) + 1)

    plt.yscale('log')  # Change to log scale
    plt.ylim(1, max_pulls * 10)  # Set consistent Y-axis limits for all plots

    # Save frame as PNG in tmp_frames_dir
    frame_filename = os.path.join(tmp_frames_dir, "file{0:05d}.png".format(idx + 1))
    plt.savefig(frame_filename, bbox_inches='tight')
    plt.close()
# - - - - - - - - - - - -

# Use GIFMaker to create GIF from frames
gif_maker = GIFMaker(delay=40)
gif_path = os.path.join(save_dir, f"{in_name}_arms_distribution.gif")
gif_maker.make_gif(gif_path, tmp_frames_dir)

gif_maker.cleanup_frames(tmp_frames_dir)
