import numpy as np
from src.utils import do_bookkeeping_cost_subsidy, simulate_bandit_rewards
from src.simulate_play.policy_library import UCB, MTR_UCB
import sys
import argparse
from src.instance_handling.get_instance import read_instance_from_file


# Command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("-file", action="store", dest="file")
parser.add_argument("-STEP", action="store", dest="STEP", type=int, default=1)
parser.add_argument("-horizon", action="store", dest="horizon", type=float, default=50000)
parser.add_argument("-nruns", action="store", dest="nruns", type=int, default=50)
args = parser.parse_args()
# Get the input bandit instance file_name
in_file = args.file
# Policies to be simulated
algos = ['ucb', 'mtr-ucb']
# Horizon/ max number of iterations
horizon = int(args.horizon)
# Number of runs to average over
nruns = args.nruns
# Step interval for which data is recorded
STEP = args.STEP


if __name__ == '__main__':
    # Read the bandit instance from file
    instance_data = read_instance_from_file(in_file)
    arm_reward_array = instance_data.get('arm_reward_array', None)
    # Abort if there is no arm_reward_array
    if arm_reward_array is None:
        raise ValueError("No arm_reward_array found in the input file")
    min_reward = instance_data.get('min_reward', None)
    # Abort if there is no min_reward
    if min_reward is None:
        raise ValueError("No min_reward found in the input file")
    arm_cost_array = instance_data.get('arm_cost_array', None)
    # Abort if there is no arm_cost_array
    if arm_cost_array is None:
        raise ValueError("No arm_cost_array found in the input file")
    # Infer the number of arms from the list of rewards/costs
    n_arms = len(arm_reward_array)
    # Compute the calibration quantities needed for quality and cost regret
    # - - - - - - - - - - - - - - - -
    # Create a boolean array of arms that have reward >= min_reward
    acceptable_arms = arm_reward_array >= min_reward
    # Set costs of invalid arms to a high value
    cost_array_filter = np.where(acceptable_arms, arm_cost_array, np.inf)
    # Get the tolerated action with the minimum cost against which we shall calibrate cost regret
    k_calib = np.argmin(cost_array_filter)
    mu_calib = min_reward
    # Get the cost of the optimal arm
    c_calib = arm_cost_array[k_calib]
    # Print a column headers for the output file
    sys.stdout.write("algo,rs,time-step,qual_reg,cost_reg,nsamps\n")
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
            # For every run of every algorithm, prepend the (t = 0, regret = 0, nsamps = 0) data point
            #  to the output file
            sys.stdout.write("{0}, {1}, {2}, {3:.2f}, {4:.2f}, {5}\n".format(al, rs, 0, qual_reg, cost_reg,
                                                                             ';'.join(['0'] * n_arms)))
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
                    # Do book-keeping for this policy, and receive all the params that were modified
                    nsamps, mu_hat, qual_reg, cost_reg = (
                        do_bookkeeping_cost_subsidy(STEP=STEP, arm_samples=arm_samples, k=k, t=t, nsamps=nsamps,
                                                    mu_hat=mu_hat, qual_reg=qual_reg, cost_reg=cost_reg, al=al,
                                                    rs=rs, arm_reward_array=arm_reward_array, mu_calib=mu_calib,
                                                    arm_cost_array=arm_cost_array, c_calib=c_calib))
            elif al == 'mtr-ucb':
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Number of times a certain arm is sampled, each arm is sampled once at start
                nsamps = np.zeros(n_arms, dtype=np.int32)
                for t in range(1, horizon + 1):
                    # To initialise estimates from all arms
                    if t < n_arms + 1:
                        # sample the arm with (array) index (t - 1)
                        k = t - 1
                    else:
                        # Update ucb index value for all arms based on quantities from
                        # previous iteration and obtain arm index to sample
                        k = MTR_UCB(mu_hat, nsamps, t, arm_cost_array, min_reward)
                    nsamps, mu_hat, qual_reg, cost_reg = (
                        do_bookkeeping_cost_subsidy(STEP=STEP, arm_samples=arm_samples, k=k, t=t, nsamps=nsamps,
                                                    mu_hat=mu_hat, qual_reg=qual_reg, cost_reg=cost_reg, al=al,
                                                    rs=rs, arm_reward_array=arm_reward_array, mu_calib=mu_calib,
                                                    arm_cost_array=arm_cost_array, c_calib=c_calib))
            else:
                print("Invalid algorithm {0} selected_algos, ignored".format(al))
                continue
