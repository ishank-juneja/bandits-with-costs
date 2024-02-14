import numpy as np
from src.utils import do_bookkeeping_conventional, simulate_bandit_rewards
import sys
import argparse
from src.simulate_play.policy_library import improved_ucb, UCB
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
algos = ['ucb', 'improved-ucb']
# Horizon/ max number of iterations
horizon = int(args.horizon)
# Number of runs to average over
nruns = args.nruns
# Step interval for which data is recorded
STEP = args.STEP


if __name__ == '__main__':
    # Read a regular/vanilla bandit instance with only rewards from file
    instance_data = read_instance_from_file(in_file)
    arm_reward_array = instance_data.get('arm_reward_array', None)
    # Abort if there is no arm_reward_array
    if arm_reward_array is None:
        raise ValueError("No arm_reward_array found in the input file")
    # Infer the number of arms from the list of rewards/costs
    n_arms = len(arm_reward_array)
    # Get the action against which we calibrate reward (here best arm)
    k_calib = np.argmax(arm_reward_array)
    # Get the expected reward of the best arm
    mu_calib = arm_reward_array[k_calib]
    # Print a column headers for the output file
    sys.stdout.write("algo,rs,horizon,reg,nsamps\n")
    for al in algos:
        for rs in range(nruns):
            # Set numpy random seed to make output deterministic for a given run
            np.random.seed(rs)
            arm_samples = simulate_bandit_rewards(arm_reward_array, horizon)
            # Initialize cumulative expected regret
            # Expected cost regret
            reg = 0.0
            # For every run of every algorithm, prepend the (t = 0, regret = 0, nsamps = 0) data point
            #  to the output file
            sys.stdout.write("{0}, {1}, {2}, {3:.2f}, {4}\n".format(al, rs, 0, reg, ';'.join(['0'] * n_arms)))
            if al == 'ucb':
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Number of times a certain arm is sampled, each arm is sampled once at start
                nsamps = np.zeros(n_arms, dtype=np.int32)
                # Begin policy loop
                for t in range(1, horizon + 1):
                    # To initialise estimates from all arms
                    if t < n_arms + 1:
                        # sample the arm with (array) index (t - 1)
                        k = t - 1
                    else:
                        # Pass the latest params to the policy and get the arm index to sample
                        k = UCB(mu_hat, nsamps, t)
                    # Do book-keeping for this policy, and receive all the params that were modified
                    nsamps, mu_hat, reg = do_bookkeeping_conventional(STEP=STEP, arm_samples=arm_samples, k=k, t=t, nsamps=nsamps,
                                                                      mu_hat=mu_hat, reg=reg, al=al, rs=rs,
                                                                      arm_reward_array=arm_reward_array, mu_opt=mu_calib)
            elif al == 'improved-ucb':
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Number of times a certain arm is sampled, each arm is sampled once at start
                nsamps = np.zeros(n_arms, dtype=np.int32)
                # Create and initialize variables to track delta_tilde, and B_0
                delta_tilde = 1.0
                # Create a list out of the indices of all the arms
                # We always ensure that any updated B remains sorted
                B = list(range(n_arms))
                # Variable to hold the arm in B_m that was most recently sampled
                #  since the way the algorithm works, we need to keep sampling the same arm
                #  until we have sampled it n_m times
                last_sampled = 0
                # Begin policy loop
                for t in range(1, horizon + 1):
                    # Since the algorithm works in a phased/batched way, the quantities delta_tilde and B
                    #  will be updated occasionally like a step function
                    k, delta_tilde, B = improved_ucb(mu_hat, nsamps, horizon, delta_tilde, B, last_sampled)
                    # Update the last sampled arm index
                    last_sampled = k
                    # Do book-keeping for this policy, and receive all the params that were modified
                    nsamps, mu_hat, reg = do_bookkeeping_conventional(STEP=STEP, arm_samples=arm_samples, k=k, t=t, nsamps=nsamps,
                                                                      mu_hat=mu_hat, reg=reg, al=al, rs=rs,
                                                                      arm_reward_array=arm_reward_array, mu_opt=mu_calib)
            else:
                raise ValueError(f"Unknown algorithm: {al}")
