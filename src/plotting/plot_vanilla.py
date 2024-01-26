import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import argparse
import pathlib


# Command line input
parser = argparse.ArgumentParser()
parser.add_argument("-file", action="store", dest="file")
parser.add_argument("-STEP", action="store", dest="STEP", type=int)
parser.add_argument("-horizon", action="store", dest="horizon", type=float)
args = parser.parse_args()
in_file = args.file
# Extract out just file name sans extension
in_name = in_file.split('/')[-1].split('.')[0]
# Ignore last 4 characters _log
in_name = in_name[:-4]
# Create folder to save results if it doesn't already exist
pathlib.Path('results/plots/' + in_name).mkdir(parents=False, exist_ok=True)
x_points = np.arange(args.STEP, int(args.horizon) + 1, args.STEP)
y_points = np.zeros_like(x_points)
selected = ['ucb', 'improved-ucb']
# Number of distinct algorithms used
nalgos = len(selected)
# Number of colors should be at least as many as number of LABELS
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
# Choose as many colors as there are algorithms selected_algos
if len(COLORS) < len(selected):
	print("Error: Not enough colors present for plotting\nAdd colors or reduce number of algorithms")
	exit(-1)
else:
	COLORS = COLORS[:len(selected)]

bandit_data = pd.read_csv(in_file, sep=",", header=None)
bandit_data.columns = ["algo", "rs", "horizon", "reg", "nsamps"]
# List of dependent variables to be plotted, from above list
# Dependents are Cost Regret, Quality Regret
scalar_dependents = bandit_data.columns[3:-1]	# Exclude the last column which consists of

# Plot the regret data
for dependent in scalar_dependents:
    plt.figure(figsize=(10, 10))
    for index, label in enumerate(selected):
        cur_data = bandit_data.loc[bandit_data["algo"] == label]
        for i in range(len(y_points)):
            y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
        plt.plot(x_points, y_points, color=COLORS[index], linewidth=3)

    plt.legend(selected, fontsize='large')  # Increase font size for legend
    plt.xlabel("Number of rounds T", fontweight="bold", fontsize=14)  # Increase font size for x-axis label
    plt.ylabel("Cumulative Regret", fontweight="bold", fontsize=14)  # Increase font size for y-axis label
    plt.title("CS Problem Policy Comparisons", fontweight="bold")
    plt.yticks()
    plt.savefig('results/plots/' + in_name + "/{0}_{1}_complete_plot".format(in_name, dependent) + ".png", bbox_inches="tight")
    plt.close()
