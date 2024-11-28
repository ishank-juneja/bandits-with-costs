import argparse
import matplotlib.pyplot as plt
from src.instance_handling import read_instance_from_file

# Command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--file", action="store", dest="file")
parser.add_argument('--save-dir', type=str, help='The directory to save the plots in.',
                    default="data/bandit_instances/full_cost_subsidy")
args = parser.parse_args()
# Get the input bandit instance file_name
in_file = args.file
# Extract out just file name sans extension
in_name = in_file.split('/')[-1].split('.')[0]

if __name__ == '__main__':
    # Read the bandit instance from file
    instance_data = read_instance_from_file(in_file)
    arm_reward_array = instance_data.get('arm_reward_array', None)
    if arm_reward_array is None:
        raise ValueError("No arm_reward_array found in the input file")
    subsidy_factor = instance_data.get('subsidy_factor', None)
    if subsidy_factor is None:
        raise ValueError("No subsidy_factor found in the input file")
    arm_cost_array = instance_data.get('arm_cost_array', None)
    if arm_cost_array is None:
        raise ValueError("No arm_cost_array found in the input file")

    # Get the instance ID
    instance_id = instance_data.get('instance_id', None)

    # Infer the number of arms from the list of rewards/costs
    n_arms = len(arm_reward_array)
    arm_indices = range(n_arms)

    plt.figure(figsize=(12, 6))
    for idx in arm_indices:
        plt.plot([idx, idx], [0, arm_reward_array[idx] / 2 - 0.02], 'k-', lw=1)
        plt.plot([idx, idx], [arm_reward_array[idx] / 2 + 0.1, arm_reward_array[idx]], 'k-', lw=1)
        start_x = idx - 0.2
        end_x = idx + 0.2
        y_value = arm_reward_array[idx]
        plt.plot([start_x, end_x], [y_value, y_value], 'b-', lw=1)
        # First line with the variable name and equal sign
        plt.text(idx, arm_reward_array[idx] + 0.04,
                 r"$c_{{{0}}}$".format(idx + 1),
                 ha='center', va='bottom', color='black', fontweight='bold', fontsize=12)
        # Second line with the value
        plt.text(idx, arm_reward_array[idx],  # Adjust vertical position as needed
                 r"${0:.2f}$".format(arm_cost_array[idx]),
                 ha='center', va='bottom', color='black', fontweight='bold', fontsize=11)
        # Variable name on the first line
        plt.text(idx, (arm_reward_array[idx]) / 2 + 0.04,  # Slightly raised to make room for the second line
                 r"$\mu_{{{0}}}$".format(idx + 1),
                 ha='center', va='bottom', color='black', fontweight='bold', fontsize=12)
        # Value on the second line
        plt.text(idx, (arm_reward_array[idx]) / 2,  # Original position for the value
                 r"${0:.2f}$".format(arm_reward_array[idx]),
                 ha='center', va='bottom', color='black', fontweight='bold', fontsize=11)

    # Infer best reward from the arm reward array
    # best_reward = max(arm_reward_array)
    # # Compute \mu_\CS = (1 - \alpha) \mu^* for the specified values of alpha
    # alphas = [0.05, 0.1, 0.15, 0.2]
    # mu_cs_values = [(1 - alpha) * best_reward for alpha in alphas]

    # Plot each mu_CS value as a horizontal line
    # for mu_cs_value in mu_cs_values:
    #     plt.axhline(y=mu_cs_value, color='red', linestyle='--', linewidth=1)  # Updated linestyle

    plt.ylim(top=1.1)
    plt.title(f"Goodreads Experiment", fontsize=18, fontweight='bold')
    plt.xlabel("Arm Index", fontsize=16)
    plt.ylabel("Expected Quality", fontsize=16)
    plt.xticks(range(n_arms), [f"{i + 1}" for i in range(n_arms)], fontsize=16)
    plt.yticks(fontsize=16, fontweight='bold')

    # Get limits of the plot along the x-axis
    x_min, x_max = plt.xlim()
    extreme_right_x = x_max * 1.01
    # for idx, mu_cs_value in enumerate(mu_cs_values):
    #     # plt.text(extreme_right_x, mu_cs_value, r"$\mu_{{\text{{CS}}}} = {0:.2f}$".format(mu_cs_value),
    #     #          ha='left', va='center', color='red', fontweight='bold', fontsize=12)
    #     plt.text(extreme_right_x, mu_cs_value, r"$\alpha = {0:.2f}$".format(alphas[idx]),
    #              ha='left', va='center', color='red', fontweight='bold', fontsize=12)

    # Retrieve the path for the directory to save the plots in
    save_dir = args.save_dir
    # Save figure
    plt.savefig(save_dir + "/{0}".format(in_name) + ".pdf", bbox_inches="tight")
    plt.close()
