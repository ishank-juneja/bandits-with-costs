import numpy as np


def read_instance_from_file(file_path):
    with open(file_path, 'r') as file:
        data = {}
        for line in file:
            if not line.startswith('#') and line.strip():
                key, value = line.split(':')
                value = np.array([float(x.strip()) for x in value.split(',')])
                data[key.strip()] = value
        return data


if __name__ == '__main__':
    # Example usage
    file_path = '../../data/bandit_instances/I1.txt'
    instance_data = read_instance_from_file(file_path)

    arm_reward_array = instance_data.get('arm_reward_array', None)
    min_reward = instance_data.get('min_reward', None)[0]
    arm_cost_array = instance_data.get('arm_cost_array', None)

    print(arm_reward_array)
    print(min_reward)
    print(arm_cost_array)
