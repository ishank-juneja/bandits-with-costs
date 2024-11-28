import json
import os
import numpy as np

def process_json_files_and_create_bandit_instance(folder_path, output_file_path):
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    num_files = len(json_files)

    # Initialize arrays for arm rewards and costs
    arm_rewards = np.zeros(num_files)
    arm_costs = np.random.rand(num_files)  # Random costs for each arm

    # Process each file and calculate the average rating
    for idx, file in enumerate(json_files):
        total_ratings = 0
        line_count = 0
        full_path = os.path.join(folder_path, file)
        with open(full_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    clipped_rating = 1.0 if int(data['rating']) > 3 else 0.0
                    total_ratings += clipped_rating
                    line_count += 1
                except json.JSONDecodeError:
                    print("Error decoding JSON")
                except KeyError:
                    print("Key not found in JSON line")

        if line_count > 0:
            average_rating = total_ratings / line_count
            arm_rewards[idx] = average_rating
            print(f'{os.path.basename(full_path)}:\t{average_rating:.2f}\t{line_count}\tCost: {arm_costs[idx]:.2f}')
        else:
            print('No valid entries found.')

    # Sort by arm costs in ascending order and reorder arm rewards accordingly
    sorted_indices = np.argsort(arm_costs)
    sorted_arm_costs = arm_costs[sorted_indices]
    sorted_arm_rewards = arm_rewards[sorted_indices]

    # Save to file
    instance_id = "GR001"
    subsidy_factor = 0.05
    content = f"instance_id: {instance_id}\n"
    content += f"arm_reward_array: {', '.join(f'{reward:.3f}' for reward in sorted_arm_rewards)}\n"
    content += f"subsidy_factor: {subsidy_factor}\n"
    content += f"arm_cost_array: {', '.join(f'{reward:.3f}' for reward in sorted_arm_costs)}\n"


    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as file:
        file.write(content)

# Example usage
folder_path = "data/good_reads"
output_file_path = "data/bandit_instances/good_reads_instance.txt"
process_json_files_and_create_bandit_instance(folder_path, output_file_path)
