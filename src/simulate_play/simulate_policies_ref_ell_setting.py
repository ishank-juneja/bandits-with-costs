import numpy as np
from src.utils import do_bookkeeping_cost_subsidy, simulate_bandit_rewards
import sys
import argparse
from src.policy_library import improved_ucb, asymmetric_pe, pairwise_elimination, cs_ucb_known_ell, UCB
from src.instance_handling.get_instance import read_instance_from_file


# Command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("-file", action="store", dest="file")
parser.add_argument("-STEP", action="store", dest="STEP", type=int, default=10000)
parser.add_argument("-horizon", action="store", dest="horizon", type=float, default=int(5e6))
parser.add_argument("-nruns", action="store", dest="nruns", type=int, default=5)
args = parser.parse_args()
# Get the input bandit instance file_name
in_file = args.file
# Policies to be simulated
algos = ['pe', 'asymmetric-pe', 'ucb-cs']

# algos = ['asymmetric-pe']
# Horizon/ max number of iterations
horizon = int(args.horizon)
# Number of runs to average over
nruns = args.nruns
# Step interval for which data is recorded
STEP = args.STEP


def prune_arms(arm_cost_array: np.array, ref_arm_ell: int):
    """
    Prune arms with cost strictly higher than arm ell by finding the
    smallest index of the array arm_cost_array where the cost becomes
    strictly greater than arm_ell_cost
    :param arm_cost_array: Array of length K
    :param ref_arm_ell:
    :return: The index of the last element
    """
    # Retrieve the cost of the reference arm
    arm_ell_cost = arm_cost_array[ref_arm_ell]
    # Iterate through each index and element in the array
    for idx, cost in enumerate(arm_cost_array):
        # Check if the current element is strictly greater than arm_ell_cost
        if cost > arm_ell_cost:
            return idx
    # Return the length of the array since the whole set of arms is to be included
    return len(arm_cost_array)


if __name__ == '__main__':
    # Read a bandit instance designed for the known
    #  reference arm ell setting from file
    instance_data = read_instance_from_file(in_file)
    arm_reward_array = instance_data.get('arm_reward_array', None)
    if arm_reward_array is None:
        raise ValueError("No arm_reward_array found in the input file")
    arm_cost_array = instance_data.get('arm_cost_array', None)
    if arm_cost_array is None:
        raise ValueError("No arm_cost_array found in the input file")
    ref_arm_ell = instance_data.get('ref_arm_ell', None)
    if ref_arm_ell is None:
        raise ValueError("No ref_arm_ell found in the input file")
    arm_ell_cost = arm_cost_array[ref_arm_ell]
    # Infer the nominal number of arms without any pre-processing/pruning
    n_arms = len(arm_reward_array)

    # Determine the expected return of the reference arm to calibrate quality regret
    mu_calib = arm_reward_array[ref_arm_ell]

    # Determine the cost of the best action to calibrate cost regret
    # Set the cost of all the un-acceptable arms to a high value
    acceptable_arms = arm_reward_array >= mu_calib
    # Set costs of invalid arms to a high value
    cost_array_filter = np.where(acceptable_arms, arm_cost_array, np.inf)
    # Get the acceptable action with the minimum cost against which we shall calibrate cost regret
    k_calib = np.argmin(cost_array_filter)
    # Get the cost of the best action
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
            if al == 'pe':
                # Preprocessing
                # - - - - - - - -
                # Perform pruning by removing arms with cost strictly higher than arm ell
                # Get the smallest index of the array arm_cost_array where the cost becomes
                #  strictly greater than arm_ell_cost, and then only include arms before that index
                min_idx = prune_arms(arm_cost_array, ref_arm_ell)
                # Update the number of arms for this algorithm based on this pruning
                n_arms_pruned = min_idx
                # Update arm_reward_array and arm_cost_array based on this pruning
                arm_reward_array = arm_reward_array[:min_idx]
                arm_cost_array = arm_cost_array[:min_idx]
                # Update the arm_samples to be used to simulate rewards
                arm_samples = arm_samples[:min_idx, :]
                # - - - - - - - -

                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms_pruned)
                # Number of times a certain arm is sampled, each arm is sampled once at start
                nsamps = np.zeros(n_arms, dtype=np.int32)
                # Create and initialize variables to track delta_tilde
                delta_tilde = 1.0
                # The last sampled arm is useful information to make a quick decision about
                #  the arm to be sampled in the very next round
                last_sampled = 0
                # Variable to track episode number and by extension the candidate arm being processed
                # Updated by the pairwise elimination algorithm function call
                episode = 0
                # Begin policy loop
                for t in range(1, horizon + 1):
                    # Receive the arm index to sample, and the updated delta_tilde, and episode number
                    k, delta_tilde, episode = pairwise_elimination(ref_ell_idx=ref_arm_ell , mu_hat=mu_hat,
                                                                   nsamps=nsamps, horizon=horizon,
                                                                   delta_tilde=delta_tilde, episode_num=episode,
                                                                   last_sampled=last_sampled)
                    # Update the last sampled arm index
                    last_sampled = k
                    # Do book-keeping for this policy, and receive all the params that were modified
                    nsamps, mu_hat, qual_reg, cost_reg = (
                        do_bookkeeping_cost_subsidy(STEP=STEP, arm_samples=arm_samples, k=k, t=t, nsamps=nsamps,
                                                    mu_hat=mu_hat, qual_reg=qual_reg, cost_reg=cost_reg, al=al,
                                                    rs=rs, arm_reward_array=arm_reward_array, mu_calib=mu_calib,
                                                    arm_cost_array=arm_cost_array, c_calib=c_calib))
            elif al == 'asymmetric-pe':
                # Preprocessing (identical to regular pe)
                # - - - - - - - -
                # Perform pruning by removing arms with cost strictly higher than arm ell
                # Get the smallest index of the array arm_cost_array where the cost becomes
                #  strictly greater than arm_ell_cost, and then only include arms before that index
                min_idx = prune_arms(arm_cost_array, ref_arm_ell)
                # Update the number of arms for this algorithm based on this pruning
                n_arms_pruned = min_idx
                # Update arm_reward_array and arm_cost_array based on this pruning
                arm_reward_array = arm_reward_array[:min_idx]
                arm_cost_array = arm_cost_array[:min_idx]
                # Update the arm_samples to be used to simulate rewards
                arm_samples = arm_samples[:min_idx, :]
                # - - - - - - - -

                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms_pruned)
                # Number of times a certain arm is sampled, each arm is sampled once at start
                nsamps = np.zeros(n_arms, dtype=np.int32)
                # Create and initialize variables to track delta_tilde_ell and delta_tilde
                delta_tilde_ell = 1.0
                delta_tilde = 1.0
                # The last sampled arm is useful information to make a quick decision about
                #  the arm to be sampled in the very next round
                last_sampled = 0
                # Variable to track episode number and by extension the candidate arm being processed
                # Updated by the pairwise elimination algorithm function call
                episode = 0
                # Begin policy loop
                for t in range(1, horizon + 1):
                    # Receive the arm index to sample, and the updated delta_tilde, and episode number
                    k, delta_tilde_ell, delta_tilde, episode = asymmetric_pe(ref_ell_idx=ref_arm_ell,
                                                                             mu_hat=mu_hat, nsamps=nsamps,
                                                                             horizon=horizon,
                                                                             delta_tilde_ell=delta_tilde_ell,
                                                                             delta_tilde=delta_tilde,
                                                                             episode_num=episode,
                                                                             last_sampled=last_sampled)
                    # Update the last sampled arm index
                    last_sampled = k
                    # Do book-keeping for this policy, and receive all the params that were modified
                    nsamps, mu_hat, qual_reg, cost_reg = (
                        do_bookkeeping_cost_subsidy(STEP=STEP, arm_samples=arm_samples, k=k, t=t, nsamps=nsamps,
                                                    mu_hat=mu_hat, qual_reg=qual_reg, cost_reg=cost_reg, al=al,
                                                    rs=rs, arm_reward_array=arm_reward_array, mu_calib=mu_calib,
                                                    arm_cost_array=arm_cost_array, c_calib=c_calib))
            elif al == 'ucb-cs':
                # Preprocessing (identical to regular pe)
                # - - - - - - - -
                # Perform pruning by removing arms with cost strictly higher than arm ell
                # Get the smallest index of the array arm_cost_array where the cost becomes
                #  strictly greater than arm_ell_cost, and then only include arms before that index
                min_idx = prune_arms(arm_cost_array, ref_arm_ell)
                # Update the number of arms for this algorithm based on this pruning
                n_arms_pruned = min_idx
                # Update arm_reward_array and arm_cost_array based on this pruning
                arm_reward_array = arm_reward_array[:min_idx]
                arm_cost_array = arm_cost_array[:min_idx]
                # Update the arm_samples to be used to simulate rewards
                arm_samples = arm_samples[:min_idx, :]
                # - - - - - - - -
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms_pruned)
                # Number of times a certain arm is sampled, each arm is sampled once at start
                nsamps = np.zeros(n_arms_pruned, dtype=np.int32)
                # Begin policy loop
                for t in range(1, horizon + 1):
                    # Receive the arm index to sample, and the updated delta_tilde, and episode number
                    k = cs_ucb_known_ell(ref_ell_idx=ref_arm_ell,
                                         mu_hat=mu_hat, costs=arm_cost_array,
                                         t=t, nsamps=nsamps, horizon=horizon)
                    # Update the last sampled arm index
                    last_sampled = k
                    # Do book-keeping for this policy, and receive all the params that were modified
                    nsamps, mu_hat, qual_reg, cost_reg = (
                        do_bookkeeping_cost_subsidy(STEP=STEP, arm_samples=arm_samples, k=k, t=t, nsamps=nsamps,
                                                    mu_hat=mu_hat, qual_reg=qual_reg, cost_reg=cost_reg, al=al,
                                                    rs=rs, arm_reward_array=arm_reward_array, mu_calib=mu_calib,
                                                    arm_cost_array=arm_cost_array, c_calib=c_calib))

            else:
                raise ValueError(f"Unknown algorithm: {al}")

