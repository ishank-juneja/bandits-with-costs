import numpy as np
from math import ceil, log, sqrt


def improved_ucb(mu_hat, nsamps, horizon, delta_tilde, B, last_sampled):
    """
    A function that implements the Improved UCB algorithm.
    https://www.overleaf.com/project/6502fd4306f4b073aa6bd809
    ***********************
    In every iteration of this function, we must sample an arm
    Case 1: A single active arm is left in B_m
    if so then we must sample that arm until horizon is exhausted
    Case 2: Multiple arms are left in B_m and a round is ongoing
    if so we check if the most recently sampled arm has been sampled n_m times
    if not, we sample that arm again
    if yes, we move to the arm with the next highest index in B_m if there is one
    if there is no next arm in B_m, we end the round, and move to arm elimination ...
    While eliminating arms, we use the criteria for eliminating an arm as given in the paper
    and update the round number m, delta_tilde, and B_m accordingly
    ***********************
    mu_hat: Empirical estimates of rewards for each arm
    nsamps: Number of times each arm has been sampled
    horizon: Known horizon as input
    delta_tilde: Gaps used by the elimination to set number of samples in a batch and
     UCB buffer terms
    B: List of arms that are not yet eliminated
    last_sampled: Index of the last sampled arm so that we know which arm to sampled next
     in batched/rounded sampling
    """
    # Check if there is only one arm left in B_m
    if len(B) == 1:
        # If so, keep returning that arm until the calling horizon for loop gets terminated
        k = B[0]
        # Returned parameters other than k are effectively don't cares
        return k, delta_tilde, B
    # Else if there is more than one arm and an ongoing round m, then keep
    #  batch sampling arms until n_m samples have been collected for each of them
    elif len(B) > 1:
        # Recompute n_m for the current round m
        n_m = ceil(2 * log(horizon * delta_tilde ** 2) / (delta_tilde ** 2) )
        # Check the number of times the last_sampled arm has been sampled
        if nsamps[last_sampled] < n_m:
            # If it has not been sampled n_m times, return it again with
            #  all other parameters unchanged
            return last_sampled, delta_tilde, B
        elif nsamps[last_sampled] == n_m:
            # If sampled n_m times, move to the next arm in B_m
            #  If there is no next arm, end the round
            if last_sampled == B[-1]:
                # Round is complete, move to arm elimination phase
                pass
            else:
                # Else, move to the next arm in B_m
                k = B[B.index(last_sampled) + 1]
                # Return all other parameters unchanged
                return k, delta_tilde, B
    # Start arm elimination phase
    # Compute the buffer terms for UCB/LCB indices
    buffer = sqrt( log(horizon * delta_tilde ** 2) / (2 * n_m) )
    # Using mean and buffer, compute the LCB of all the active arms
    active_lcb_list = [mu_hat[k] - buffer for k in B]

    # Initialize a new list to hold the indices of the arms to keep
    B_new = []

    for arm_indices_k in B:
        # Compute the UCB of arm k
        ucb_k = mu_hat[arm_indices_k] + buffer
        # Check if the UCB of arm k is under the LCB of any other arm
        if ucb_k >= max(active_lcb_list):
            # If not, add arm k to the new list
            B_new.append(arm_indices_k)

    # Replace the old list with the new list
    B = B_new

    # Update delta_tilde
    delta_tilde = delta_tilde / 2
    # Return package says sample the lowest index arm in the set of active arms
    k = B[0]
    # Return all other parameters
    return k, delta_tilde, B


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
