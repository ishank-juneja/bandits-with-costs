import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from matplotlib.lines import Line2D

# Hardcoded folders
folders = [
    "results/_run_logs/ml_known_ell_vary_ell",
    "results/_run_logs/gr_known_ell_vary_ell",
    "results/_run_logs/ml_known_ell_vary_alpha",
    "results/_run_logs/gr_known_ell_vary_alpha",
]
save_dir = 'results/plots/trade_off'

selected_algos = ['pe', 'ucb-cs']
# save_dir = 'path/to/save/dir'  # Example save directory

# Map algo names used in the log files to algo names used in the code
custom_algo_names = {
    'cs-pe': 'pe-cs',
    'cs-ucb': 'ucb-cs',
    'cs-ts': 'ts-cs',
    'cs-etc': 'etc-cs',
    'pe': 'pe',
    'ucb-cs': 'ucb-cs'
}

# Marker styles for different algorithms
marker_styles = ['o', 's', '^', '*']
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# Plotting setup
fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=False, sharey=False)
fig.suptitle('Tradeoff in Known Reference Arm Setting', fontsize=18, fontweight='bold', y=1.02)

# Applying logarithmic scale and setting labels
for ax in axs.flat:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Terminal Quality Regret', fontsize=16, fontweight='bold')
    ax.set_ylabel('Terminal Cost Regret', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)


# Process each folder
for idx, folder in enumerate(folders):
    sorted_files_folder = sorted(pathlib.Path(folder).iterdir(), key=lambda x: x.name)
    nseeds = -1

    for file in sorted_files_folder:
        bandit_data = pd.read_csv(file, sep=",")

        if nseeds == -1:
            nseeds = bandit_data["rs"].max() + 1
        elif nseeds != bandit_data["rs"].max() + 1:
            raise ValueError("Number of seeds mismatch in log files")

        for algo in selected_algos:
            algo_data = bandit_data[bandit_data['algo'] == algo]
            horizon = algo_data['time-step'].max()
            quality_regret = algo_data[algo_data['time-step'] == horizon]['qual_reg']
            cost_regret = algo_data[algo_data['time-step'] == horizon]['cost_reg']
            # Check if any zeros are present in the cost/quality regret, make them 1.0 instead for log scale
            cost_regret = cost_regret.replace(0, 1.0)
            quality_regret = quality_regret.replace(0, 1.0)
            ax = axs[idx // 2, idx % 2]
            ax.scatter(quality_regret, cost_regret, label=custom_algo_names[algo], color=COLORS[selected_algos.index(algo)], marker=marker_styles[selected_algos.index(algo)], alpha=0.5)

# Define a custom legend for one of the plots
legend_elements = [
    Line2D([0], [0], marker=marker_styles[i], color='w', label=custom_algo_names[selected_algos[i]], markerfacecolor=COLORS[i], markersize=10)
    for i in range(len(selected_algos))
]
axs[1, 1].legend(handles=legend_elements, loc='lower right', fontsize=14)

# Layout adjustment
plt.tight_layout()

# Display the plot
# plt.show()

# Optionally save the figure
plt.savefig(save_dir + "/tradeoff_ref_ell.pdf", bbox_inches="tight")
plt.close()