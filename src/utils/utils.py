import numpy as np
import sys


def do_bookkeeping_conventional(STEP, arm_samples, k, t, nsamps, mu_hat, reg, al, rs, arm_reward_array, mu_opt):
    """
    STEP: The step-interval at which data is recorded
    arm_samples: NumPy array of shape (n_arms, horizon) of pre-sampled rewards for each arm at each time step.
    k: The arm index to sample at time t
    t: The current time step
    nsamps: NumPy array of shape (n_arms,) containing the number of times each arm has been sampled.
    mu_hat: NumPy array of shape (n_arms,) containing the empirical mean reward of each arm.
    reg: The cumulative regret incurred so far.
    al: The algorithm being simulated in string name
    rs: The random seed being used for this run
    arm_reward_array: np array containing the true mean reward of all the arms
    mu_opt: The expected return of the best arm
    return: The updated nsamps, mu_hat, (cumulative) reg (All the parameters that were modified)
    """
    # Get 0/1 reward based on arm/channel choice
    # Indexing starts from 0, so subtract 1 from t
    r = arm_samples[k, t - 1]
    # Increment number of times kth arm sampled
    nsamps[k] = nsamps[k] + 1
    # Update empirical reward estimates, compute new empirical mean
    mu_hat[k] = ((nsamps[k] - 1) * mu_hat[k] + r) / nsamps[k]
    # Get the incremental expected regret
    reg_incr = mu_opt - arm_reward_array[k]
    # Update the expected regret
    reg += reg_incr
    # Record data at intervals of STEP in file
    if t % STEP == 0:
        # Convert nsamps array to a string for CSV output
        nsamps_str = ';'.join(map(str, nsamps))

        # Writing to standard output (you might want to write to a file instead)
        sys.stdout.write(
            "{0}, {1}, {2}, {3:.2f}, {4}\n".format(
                al, rs, t, reg, nsamps_str
            )
        )
    # Return all the params that were modified
    return nsamps, mu_hat, reg


def do_bookkeeping_cost_subsidy(STEP, arm_samples, k, t, nsamps, mu_hat, qual_reg, cost_reg, al, rs,
                                arm_reward_array, mu_calib, arm_cost_array, c_calib):
    """
    STEP: The step-interval at which data is recorded
    arm_samples: NumPy array of shape (n_arms, horizon) of pre-sampled rewards for each arm at each time step.
    k: The arm index to sample at time t
    t: The current time step
    nsamps: NumPy array of shape (n_arms,) containing the number of times each arm has been sampled.
    mu_hat: NumPy array of shape (n_arms,) containing the empirical mean reward of each arm.
    qual_reg: The cumulative quality regret incurred so far.
    cost_reg: The cumulative cost regret incurred so far.
    al: The algorithm being simulated in string name
    rs: The random seed being used for this run
    arm_reward_array: np array containing the true mean reward of all the arms
    mu_calib: The expected return that we calibrate against
    arm_cost_array: np array containing the true cost of all the arms
    c_calib: The cost of the best action that we calibrate against
    return: The updated nsamps, mu_hat, qual_reg, cost_reg (Params that were modified)
    """
    # Get 0/1 reward based on arm/channel choice
    # Indexing starts from 0, so subtract 1 from t
    r = arm_samples[k, t - 1]
    # Increment number of times kth arm sampled
    nsamps[k] = nsamps[k] + 1
    # Update empirical reward estimates, compute new empirical mean
    mu_hat[k] = ((nsamps[k] - 1) * mu_hat[k] + r) / nsamps[k]
    qual_reg_incr = max(0, mu_calib - arm_reward_array[k])
    qual_reg += qual_reg_incr
    cost_reg_incr = max(0, arm_cost_array[k] - c_calib)
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
    # Return all the params that were modified
    return nsamps, mu_hat, qual_reg, cost_reg


def simulate_bandit_rewards(means, horizon):
    # Number of arms
    K = len(means)

    # Initialize an empty array for the simulated rewards
    rewards = np.zeros((K, horizon))

    # Simulate rewards for each arm
    for k in range(K):
        rewards[k, :] = np.random.binomial(1, means[k], horizon)

    return rewards

