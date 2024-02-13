import numpy as np


def read_instance_from_file(file_path):
    """
    Create a dictionary of instance data from the MAB-CS bandit instance file at file_path.
    Ensures that required fields are present and handles optional fields with defaults.
    """
    with open(file_path, 'r') as file:
        data = {}
        required_keys = {'instance_id', 'arm_reward_array'}
        for line in file:
            if not line.startswith('#') and line.strip():
                key, value = line.split(':')
                key = key.strip()

                if key == 'instance_id':
                    data[key] = value.strip()
                elif key in ['arm_reward_array', 'arm_cost_array']:
                    data[key] = np.array([float(x.strip()) for x in value.split(',')])
                elif key in ['min_reward', 'subsidy_factor']:
                    data[key] = float(value.strip())
                elif key == 'ref_arm_ell':
                    data[key] = int(value.strip())

        # Check for required keys
        if not required_keys.issubset(data.keys()):
            missing_keys = required_keys - set(data.keys())
            raise ValueError(f"Missing required key(s): {', '.join(missing_keys)}")

        # Assert that arm_cost_array is sorted in ascending order
        if 'arm_cost_array' in data:
            if not np.all(np.diff(data['arm_cost_array']) >= 0):
                raise ValueError("Arm cost array is not sorted in ascending order, rearrange bandit arms "
                                 "in instance file {0}.".format(file_path))

        # Assert that arm_reward_array and arm_cost_array have the same length
        if 'arm_cost_array' in data and len(data['arm_reward_array']) != len(data['arm_cost_array']):
            raise ValueError("Arm reward array and arm cost array have different lengths in instance file {0}."
                             .format(file_path))

        return data


if __name__ == '__main__':
    # Example 1
    file_path = 'data/bandit_instances/I1.txt'
    instance_data = read_instance_from_file(file_path)

    instance_id = instance_data['instance_id']  # Required
    arm_reward_array = instance_data['arm_reward_array']  # Required
    min_reward = instance_data.get('min_reward', 0.0)  # Optional, default to 0.0
    arm_cost_array = instance_data.get('arm_cost_array', np.array([]))  # Optional, default to empty array
    ref_arm_ell = instance_data.get('ref_arm_ell', -1)  # Optional, default to -1
    subsidy_factor = instance_data.get('subsidy_factor', 0.0)  # Optional, default to 0.0

    print("Instance ID:", instance_id)
    print("Arm Reward Array:", arm_reward_array)
    print("Minimum Reward:", min_reward)
    print("Arm Cost Array:", arm_cost_array)
    print("Reference Arm (ell):", ref_arm_ell)
    print("Subsidy Factor:", subsidy_factor)

    # Print a blank line
    print()

    # Example 2
    file_path = 'data/bandit_instances/no_cost_subsidy/I3.txt'
    instance_data = read_instance_from_file(file_path)

    instance_id = instance_data['instance_id']  # Required
    arm_reward_array = instance_data['arm_reward_array']  # Required
    min_reward = instance_data.get('min_reward', 0.0)  # Optional, default to 0.0
    arm_cost_array = instance_data.get('arm_cost_array', np.array([]))  # Optional, default to empty array
    ref_arm_ell = instance_data.get('ref_arm_ell', -1)  # Optional, default to -1
    subsidy_factor = instance_data.get('subsidy_factor', 0.0)  # Optional, default to 0.0

    print("Instance ID:", instance_id)
    print("Arm Reward Array:", arm_reward_array)
    print("Minimum Reward:", min_reward)
    print("Arm Cost Array:", arm_cost_array)
    print("Reference Arm (ell):", ref_arm_ell)
    print("Subsidy Factor:", subsidy_factor)
