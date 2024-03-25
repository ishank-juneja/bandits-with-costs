import numpy as np
from src.utils import do_bookkeeping_cost_subsidy, simulate_bandit_rewards
import sys
import argparse
from src.instance_handling.get_instance import read_instance_from_file
from src.policy_library import UCB, etc_cs, pe_cs


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
# Explore then commit - CS and Pairwise Elimination CS (Ours)
algos = ['etc-cs', 'pe-cs']
# algos = ['etc-cs']
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
    subsidy_factor = instance_data.get('subsidy_factor', None)
    # Abort if there is no subsidy_factor
    if subsidy_factor is None:
        raise ValueError("No min_reward found in the input file")
    arm_cost_array = instance_data.get('arm_cost_array', None)
    # Abort if there is no arm_cost_array
    if arm_cost_array is None:
        raise ValueError("No arm_cost_array found in the input file")
    # Infer the number of arms from the list of rewards/costs
    n_arms = len(arm_reward_array)
    # Compute the calibration quantities needed for quality and cost regret
    # - - - - - - - - - - - - - - - -
    # Create a boolean array of arms that have reward >= (1 - subsidy_factor) * max_reward
    max_reward = np.max(arm_reward_array)
    acceptable_arms = arm_reward_array >= (1 - subsidy_factor) * max_reward
    # Set costs of invalid arms to a high value
    cost_array_filter = np.where(acceptable_arms, arm_cost_array, np.inf)
    # Get the tolerated action with the minimum cost against which we shall calibrate cost regret
    k_calib = np.argmin(cost_array_filter)
    # Compute the reward against Quality regret is calculated
    mu_calib = (1 - subsidy_factor) * max_reward
    # Get the cost of the optimal arm
    c_calib = arm_cost_array[k_calib]
    # Print a column headers for the output file
    sys.stdout.write("algo,rs,time-step,qual_reg,cost_reg,nsamps\n")
    for al in algos:
        for rs in range(nruns):
            # Set numpy random seed to make output deterministic for a given run
            np.random.seed(rs)
            # Simulate/Pre-compute all the rewards we will need in this run
            arm_samples = simulate_bandit_rewards(arm_reward_array, horizon)
            # Initialize expected cumulative cost and quality regret
            qual_reg = 0.0
            cost_reg = 0.0
            # For every run of every algorithm, prepend the (t = 0, qual_reg = 0, cost_reg = 0, nsamps = 0) data point
            #  to the output file
            sys.stdout.write("{0}, {1}, {2}, {3:.2f}, {4:.2f}, {5}\n".format(al, rs, 0, qual_reg, cost_reg,                                                           ';'.join(['0'] * n_arms)))
            if al == 'etc-cs':
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Array to hold how many times a certain arm is sampled
                nsamps = np.zeros(n_arms, dtype=np.int32)
                # Compute the value of tau based on the horizon and number of arms
                tau = int(np.ceil(horizon / n_arms) ** (2 / 3))
                # Initialize last_sampled with index 0 arm
                last_sampled = 0
                for t in range(1, horizon + 1):
                    # Get arm to be sampled per the ETC-CS policy
                    k = etc_cs(mu_hat, arm_cost_array, nsamps, horizon, last_sampled, tau, alpha=subsidy_factor)
                    # Update the books and receive updated loop/algo parameters in accordance with the sampling of arm k
                    nsamps, mu_hat, qual_reg, cost_reg = do_bookkeeping_cost_subsidy(STEP=STEP, arm_samples=arm_samples,
                                                                                     k=k, t=t, nsamps=nsamps,
                                                                                     mu_hat=mu_hat, qual_reg=qual_reg,
                                                                                     cost_reg=cost_reg, al=al,
                                                                                     rs=rs,
                                                                                     arm_reward_array=arm_reward_array,
                                                                                     mu_calib=mu_calib,
                                                                                     arm_cost_array=arm_cost_array,
                                                                                     c_calib=c_calib)
            elif al == 'pe-cs':
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Array to hold how many times a certain arm is sampled
                nsamps = np.zeros(n_arms, dtype=np.int32)
                # Create and initialize variables to track delta_tilde, and B_0
                delta_tilde = 1.0 # For both imp UCB and PE
                # Create a list out of the indices of all the arms
                # We always ensure that any updated B remains sorted
                B = list(range(n_arms)) # For imp UCB
                # Variable to hold the arm in B_m that was most recently sampled
                #  since the way the improved-ucb and pe algorithms work, we need to keep sampling the same arm
                #  until we have sampled it n_m times
                last_sampled = 0    # For both imp UCB and PE
                episode = -1 # For PE, initially when in phase 1, episode is meaningless
                # Later once phase 2 starts episode numbers become meaningful
                # Begin policy loop
                for t in range(1, horizon + 1):
                    # Get arm and mutable parameters per the PE-CS policy
                    k, delta_tilde, B, episode = pe_cs(mu_hat, arm_cost_array, nsamps, horizon, last_sampled, delta_tilde,
                                                        B, episode, alpha=subsidy_factor)
                    # Update the books and receive updated loop/algo parameters in accordance with the sampling of arm k
                    nsamps, mu_hat, qual_reg, cost_reg = do_bookkeeping_cost_subsidy(STEP=STEP, arm_samples=arm_samples,
                                                                                     k=k, t=t, nsamps=nsamps,
                                                                                     mu_hat=mu_hat, qual_reg=qual_reg,
                                                                                     cost_reg=cost_reg, al=al,
                                                                                     rs=rs,
                                                                                     arm_reward_array=arm_reward_array,
                                                                                     mu_calib=mu_calib,
                                                                                     arm_cost_array=arm_cost_array,
                                                                                     c_calib=c_calib)
            else:
                print("Invalid algorithm {0} selected_algos, ignored".format(al))
                continue
