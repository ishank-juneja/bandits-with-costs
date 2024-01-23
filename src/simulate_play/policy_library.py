import numpy as np
from math import log, sqrt


def improved_ucb(horizon, arm_samples):
    """
    A function that implements the Improved UCB algorithm.
    horizon: Known horizon as input
    arm_samples: A 2D array of shape (n_arms, horizon) containing the rewards for each arm at each time step.
    """
    # Infer the number of arms from the list of rewards/costs
    n_arms = arm_samples.shape[0]
    # Array to hold empirical estimates of each arms reward expectation
    mu_hat = np.zeros(n_arms)
    # Number of times a certain arm is sampled, each arm is sampled once at start
    nsamps = np.zeros(n_arms, dtype=np.int32)
    # Now begin UCB based decisions
    for t in range(1, horizon + 1):
        # To initialise estimates from all arms
        if t < n_arms + 1:
            # sample the arm with (array) index (t - 1)
            k = t - 1
        else:
            # Update ucb index value for all arms based on quantities from
            # previous iteration and obtain arm index to sample
            k = UCB(mu_hat, nsamps, t)
        # Get 0/1 reward based on arm/channel choice
        r = arm_samples[k, t - 1]
        # Increment number of times kth arm sampled
        nsamps[k] = nsamps[k] + 1
        # Update empirical reward estimates, compute new empirical mean
        mu_hat[k] = ((nsamps[k] - 1) * mu_hat[k] + r) / nsamps[k]
        # Get the incremental expected quality regret
        qual_reg_incr = max(0, min_reward - arm_reward_array[k])
        # Get the incremental expected cost regret
        cost_reg_incr = max(0, arm_cost_array[k] - c_opt)
        # Update the expected quality regret
        qual_reg += qual_reg_incr
        # Update the expected cost regret
        cost_reg += cost_reg_incr
        # Record data at intervals of STEP in file
        if t % STEP == 0:
            # Convert nsamps array to a string for CSV output
            nsamps_str = ';'.join(map(str, nsamps))

            # Writing to standard output (you might want to write to a file instead)
            sys.stdout.write(
                "{0}, {1}, {2}, {3:.2f}, {4:.2f}, {5}\n".format(
                    al, rs, t, qual_reg, cost_reg, nsamps_str
                )
            )


def random_argmin(arr):
    """
    A wrapper around np.argmin that breaks ties between minimum index options uniformly at random.
    """
    # Find the minimum value in the array
    min_val = np.min(arr)

    # Find all indices where this minimum value occurs
    min_indices = np.where(arr == min_val)[0]

    # Select one of these indices at random and return
    return np.random.choice(min_indices)


# pestimates are empirical estimate of probabilities
# nsamps is number of times each arm is sampled
def UCB(p_estimates, nsamps, t):
    # Update ucb value
    I_ucb = p_estimates + np.sqrt(2 * log(t) / nsamps)
    # Determine arm to be sampled in current step,
    # Ties are broken by lower index preference
    k = np.argmax(I_ucb)
    return k


def MTR_UCB(p_estimates, nsamps, t, costs, theta):
    # Compute the upper confidence bounds for all arms
    I_ucb = p_estimates + np.sqrt(2 * np.log(t) / nsamps)

    # Filter arms with UCB value greater than theta
    valid_arms = np.where(I_ucb > theta)[0]

    # If no arm has UCB greater than theta, return a random arm
    if len(valid_arms) == 0:
        return np.random.choice(range(len(p_estimates)))

    # Among the valid arms, find the one with the smallest cost
    min_cost = np.min(costs[valid_arms])
    # Get all indices from the original array where the cost equals the minimum cost
    min_cost_indices = np.where(costs == min_cost)[0]

    # Select a random index from these minimum cost indices
    k = np.random.choice(min_cost_indices)

    return k


if __name__ == '__main__':
    # Example usage
    arr = np.array([1, 0, 0, 2, 0])
    random_min_index = random_argmin(arr)
    print(random_min_index)

    # Test function

    # Create a scenario for testing
    np.random.seed(4)  # Seed for reproducibility
    p_estimates = 0.1 * np.random.rand(5)  # Random success probabilities for 5 arms
    nsamps = np.array([10, 20, 15, 25, 30])  # Number of times each arm has been pulled
    t = 50  # Current time step
    costs = np.array([3, 1, 2, 5, 0.5])  # Costs for pulling each arm
    theta = 0.7  # Threshold value for UCB

    # Run the test
    selected_arm = MTR_UCB(p_estimates, nsamps, t, costs, theta)

    print("Selected arm:", selected_arm)
    print("p_estimates:", p_estimates)
    print("nsamps:", nsamps)
    print("costs:", costs)
    print("UCB values:", p_estimates + np.sqrt(2 * np.log(t) / nsamps))
