import numpy as np
from policy_library import UCB, UCB_CS
from utils import simulate_bandit_rewards
import sys
import argparse


# Command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("-STEP", action="store", dest="STEP", type=int, default=200)
parser.add_argument("-horizon", action="store", dest="horizon", type=float, default=50000)
parser.add_argument("-nruns", action="store", dest="nruns", type=int, default=50)
args = parser.parse_args()
# Policies to be simulated
algos = ['ucb', 'ucb-cs']
# Horizon/ max number of iterations
horizon = int(args.horizon)
# Number of runs to average over
nruns = args.nruns
# Step interval for which data is recorded
STEP = args.STEP


if __name__ == '__main__':
    # Create a Multi-Armed Bandit Instance by specifying the mean reward of each arm
    arm_reward_array = np.array([0.1, 0.6, 0.3, 0.8, 0.9, 0.2])
    # Specify minimum acceptable reward at each time-step (Bottom line performance)
    min_reward = 0.6
    # List of known costs for all the arms
    arm_cost_array = np.array([0.05, 0.2, 0.2, 0.4, 0.5, 0.4])
    # Assert that the number of arms and costs are the same
    assert len(arm_reward_array) == len(arm_cost_array)
    # Infer the number of arms from the list of rewards/costs
    n_arms = len(arm_reward_array)
    # Compute the quantities needed for quality and cost regret
    # Create a boolean array of arms that have reward >= min_reward
    acceptable_arms = arm_reward_array >= min_reward
    # Set costs of invalid arms to a high value
    cost_array_filter = np.where(acceptable_arms, arm_cost_array, np.inf)
    # Get the index with the minimum cost
    k_opt = np.argmin(cost_array_filter)
    # Get the return of the optimal arm
    mu_opt = arm_reward_array[k_opt]
    # Get the cost of the optimal arm
    c_opt = arm_cost_array[k_opt]
    for al in algos:
        for rs in range(nruns):
            # Set numpy random seed to make output deterministic for a given run
            np.random.seed(rs)
            arm_samples = simulate_bandit_rewards(arm_reward_array, horizon)
            # Initialize expected quality regret and expected cost regret
            # Expected quality regret
            qual_reg = 0.0
            # Expected cost regret
            cost_reg = 0.0
            # UCB: Vanilla Upper Confidence Bound Sampling algorithm
            if al == 'ucb':
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
                        sys.stdout.write(
                            "{0}, {1}, {2}, {3:.2f}, {4:.2f}\n".format(al, rs, t, qual_reg, cost_reg))
            elif al == 'ucb-cs':
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
                        k = UCB_CS(mu_hat, nsamps, t, arm_cost_array, min_reward)
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
                        sys.stdout.write(
                            "{0}, {1}, {2}, {3:.2f}, {4:.2f}\n".format(al, rs, t, qual_reg, cost_reg))
            else:
                print("Invalid algorithm {0} selected, ignored".format(al))
                continue
