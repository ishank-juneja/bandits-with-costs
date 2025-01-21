import pandas as pd
import numpy as np
import argparse
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
# For Type 3 free fonts in the figures
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Command line input
parser = argparse.ArgumentParser()
parser.add_argument("--log-file", action="store", dest="file")
parser.add_argument('--save-dir', type=str, help='The directory to save the plots in.')
args = parser.parse_args()

selected_algos = ['cs-pe', 'cs-ts', 'cs-etc']

# Map algo names used in the log files to algo names used in the writing
custom_algo_names = {
    'cs-pe': 'pe-cs',
    'cs-ucb': 'ucb-cs',
    'cs-ts': 'ts-cs',
    'cs-etc': 'etc-cs'
}

in_file = args.file
bandit_data = pd.read_csv(in_file, sep=",")
horizon = bandit_data["time-step"].max()
plot_step = bandit_data["time-step"].iloc[1] - bandit_data["time-step"].iloc[0]
bandit_data['nsamps'] = bandit_data['nsamps'].apply(lambda x: np.fromstring(x, dtype=int, sep=';'))

# Creating the figure
fig, ax = plt.subplots(figsize=(8, 6))

COLORS = ['tab:blue', 'tab:green', 'tab:red']
marker_styles = ['o', '^', '*']

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
    ax.plot(x_points, y_points, color=COLORS[idx], linewidth=3, label=custom_algo_names[algo_name],
            marker=marker_styles[idx], markersize=10, markevery=50)

ax.set_title(r'Average Regret', fontweight="bold", fontsize=24, pad=10)

# Custom formatter
def thousands_formatter(x, pos):
    return '%1.0fK' % (x * 1e-3)

def millions_formatter(x, pos):
    return '%1.0fM' % (x * 1e-6)

formatter_1K = FuncFormatter(thousands_formatter)
formatter_1M = FuncFormatter(millions_formatter)

ax.xaxis.set_major_formatter(formatter_1M)
ax.yaxis.set_major_formatter(formatter_1K)
ax.set_xlabel('Time $(t)$', fontsize=22, fontweight='bold')
ax.set_ylabel('Cost Regret + Quality Regret', fontsize=24, fontweight='bold')
ax.tick_params(axis='both', labelsize=22)
ax.grid(which='both', axis='both', color='k', alpha=0.2)

# Legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=(0.25, 0.6), ncol=2, fontsize=22,
          framealpha=1.0, handlelength=3, columnspacing=0.5)

# plt.show()

plt.savefig(f"{args.save_dir}/fig2c.pdf", bbox_inches="tight")
plt.close()
