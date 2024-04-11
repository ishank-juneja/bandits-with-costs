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
    # Abort if there is no arm_reward_array
    if arm_reward_array is None:
        raise ValueError("No arm_reward_array found in the input file")
    subsidy_factor = instance_data.get('subsidy_factor', None)
    # Abort if there is no subsidy_factor
    if subsidy_factor is None:
        raise ValueError("No min_reward found in the input file")
    arm_cost_array = instance_data.get('arm_cost_array', None)
    # Abort if there is no arm_cost_array
    if arm_cost_array is None:
        raise ValueError("No arm_cost_array found in the input file")

    # Get the instance ID
    instance_id = instance_data.get('instance_id', None)

    # Infer the number of arms from the list of rewards/costs
    n_arms = len(arm_reward_array)
    arm_indices = range(n_arms)

    # Calculating the horizontal line value
    max_reward = max(arm_reward_array)
    horizontal_line_value = (1 - subsidy_factor) * max_reward

    # Adjusting the text label position slightly above the marker for better visibility
    plt.figure(figsize=(8, 6))
    for i in arm_indices:
        plt.plot([i, i], [0, arm_reward_array[i]], 'k-', lw=1)  # Vertical line
        # Horizontal bar marker
        start_x = i - 0.2
        end_x = i + 0.2
        y_value = arm_reward_array[i]
        plt.plot([start_x, end_x], [y_value, y_value], 'b-', lw=1)
        plt.text(i, arm_reward_array[i] + 0.01, f"{arm_cost_array[i]:.2f}", ha='center', va='bottom',
                 color='black', fontweight='bold', fontsize=12)  # Adjusted Cost label

    plt.axhline(y=horizontal_line_value, color='red', linestyle='--')  # Horizontal red line

    plt.ylim(top=1.1)
    plt.title(f"Multi-Armed Bandit Instance: {instance_id}")
    plt.xlabel("Arm Index", fontsize=14)
    plt.xticks(arm_indices)
    plt.yticks(fontsize=16, fontweight='bold')

    # Get limits of the plot along the x-axis
    x_min, x_max = plt.xlim()
    extreme_right_x = x_max * 1.01
    # Add the text label for the horizontal line
    plt.text(extreme_right_x, horizontal_line_value, f"{horizontal_line_value:.2f}",
             ha='left', va='center', color='red', fontweight='bold', fontsize=12)

    # Retrieve tha path for the directory to save the plots in
    save_dir = args.save_dir
    # Save figure
    plt.savefig(save_dir + "/{0}".format(in_name) + ".png", bbox_inches="tight")
    plt.close()
