import numpy as np
from src.utils.utils import simulate_bandit_rewards
import sys
import argparse
from src.simulate_play.policy_library import improved_ucb, UCB
from src.simulate_play.get_instance import read_instance_from_file


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


def do_bookkeeping(arm_samples, k, t, nsamps, mu_hat, reg, al, rs):
    """
    arm_samples: NumPy array of shape (n_arms, horizon) of pre-sampled rewards for each arm at each time step.
    k: The arm index to sample at time t
    t: The current time step
    nsamps: NumPy array of shape (n_arms,) containing the number of times each arm has been sampled.
    mu_hat: NumPy array of shape (n_arms,) containing the empirical mean reward of each arm.
    reg: The cumulative regret incurred so far.
    al: The algorithm being simulated in string name
    rs: The random seed being used for this run
    A function to do book-keeping for all bandit algorithms, required params in the order they appear in this func.
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
    return arm_samples, nsamps, mu_hat, reg


if __name__ == '__main__':
    # Read the bandit instance from file
    instance_data = read_instance_from_file(in_file)
    arm_reward_array = instance_data.get('arm_reward_array', None)
    # Infer the number of arms from the list of rewards/costs
    n_arms = len(arm_reward_array)
    # Get the best arm
    k_opt = np.argmax(arm_reward_array)
    # Get the expected reward of the best arm
    mu_opt = arm_reward_array[k_opt]
    for al in algos:
        for rs in range(nruns):
            # Set numpy random seed to make output deterministic for a given run
            np.random.seed(rs)
            arm_samples = simulate_bandit_rewards(arm_reward_array, horizon)
            # Initialize cumulative regret for this run
            reg = 0.0
            # UCB: Vanilla Upper Confidence Bound Sampling algorithm
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
                    arm_samples, nsamps, mu_hat, reg = do_bookkeeping(arm_samples, k, t, nsamps, mu_hat, reg, al, rs)
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
                # Ctr for the round number for the improved UCB algorithm
                m = 0
                # Variable to indicate that a round is ongoing and that we should not
                #  update round related parameters yet
                ongoing_round = False
                # Variable to hold the arm in B_m that was most recently sampled
                #  since the way the algorithm works, we need to keep sampling the same arm
                #  until we have sampled it n_m times
                last_sampled = 0
                # Begin policy loop
                for t in range(1, horizon + 1):
                    # Since the algorithm works in a phased/batched way, the quantities delta_tilde and B
                    #  will be updated occasionally like a step function
                    k, m, ongoing_round, delta_tilde, B, last_sampled = improved_ucb(mu_hat, nsamps, horizon, m,
                                                                                     delta_tilde, ongoing_round, B,
                                                                                     last_sampled)
                    # Do book-keeping for this policy, and receive all the params that were modified
                    arm_samples, nsamps, mu_hat, reg = do_bookkeeping(arm_samples, k, t, nsamps, mu_hat, reg, al, rs)
            else:
                raise ValueError(f"Unknown algorithm: {al}")





















