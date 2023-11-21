import numpy as np


def simulate_bandit_rewards(means, horizon):
    # Number of arms
    K = len(means)

    # Initialize an empty array for the simulated rewards
    rewards = np.zeros((K, horizon))

    # Simulate rewards for each arm
    for k in range(K):
        rewards[k, :] = np.random.binomial(1, means[k], horizon)

    return rewards
