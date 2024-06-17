import numpy as np
from src.utils import do_bookkeeping_cost_subsidy, simulate_bandit_rewards
import sys
import argparse
from src.instance_handling.get_instance import read_instance_from_file
from src.policy_library import cs_ucb, cs_ts, cs_etc, cs_pe


# Command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("-file", action="store", dest="file")
parser.add_argument("-STEP", action="store", dest="STEP", type=int, default=50)
parser.add_argument("-horizon", action="store", dest="horizon", type=float, default=500000)
parser.add_argument("-nruns", action="store", dest="nruns", type=int, default=1)
args = parser.parse_args()
# Get the input bandit instance file_name
in_file = args.file
# Policies to be simulated
# Explore then commit - CS and Pairwise Elimination CS (Ours)
algos = ['cs-etc', 'cs-ucb', 'cs-ts', 'cs-pe']
# algos = ['cs-ucb', 'cs-ts']
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
    max_reward = np.max(arm_reward_array)   # Largest return
    # Compute the reward against Quality Regret is calibrated/calculated, called smallest tolerated reward
    mu_calib = (1 - subsidy_factor) * max_reward
    acceptable_arms = arm_reward_array >= mu_calib
    # Set costs of invalid arms to a high value
    cost_array_filter = np.where(acceptable_arms, arm_cost_array, np.inf)
    # Get the tolerated action with the minimum cost against which we shall calibrate cost regret
    k_calib = np.argmin(cost_array_filter)

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
            sys.stdout.write("{0}, {1}, {2}, {3:.2f}, {4:.2f}, {5}\n".format(al, rs, 0, qual_reg, cost_reg,
                                                                             ';'.join(['0'] * n_arms)))
            if al == 'cs-pe':
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms, dtype=np.float64)
                # Array to hold how many times a certain arm is sampled
                nsamps = np.zeros(n_arms, dtype=np.int32)
                # Array to hold the terminal round numbers achieved so far by each arm
                omega = np.zeros(n_arms, dtype=np.int32)
                # Variable to hold the index of the last sampled arm
                last_sampled = None
                # Initialize active list N to be a list with all entries 1 through n_arms
                active_list = list(range(n_arms))
                # Episode number for the pairwise elimination algorithm
                episode = 0
                # Initialize t = 1
                t = 1
                while t < horizon + 1:
                    # Get arm to be sampled per the PE-CS policy
                    k, omega_plus, active_list_plus, episode_plus = cs_pe(mu_hat, nsamps, horizon, last_sampled, omega,
                                                                          active_list, episode, alpha=subsidy_factor)
                    omega = omega_plus
                    active_list = active_list_plus
                    episode = episode_plus
                    if k is None:
                        continue
                    last_sampled = k
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
                    t += 1
            # Ablation where the round number for arm ell is not allowed to exceed the round number for arm ep
            elif al == 'cs-pe-symmetric':
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms, dtype=np.float64)
                # Array to hold how many times a certain arm is sampled
                nsamps = np.zeros(n_arms, dtype=np.int32)
                # Array to hold the terminal round numbers achieved so far by each arm
                omega = np.zeros(n_arms, dtype=np.int32)
                # Variable to hold the index of the last sampled arm
                last_sampled = None
                # Initialize active list N to be a list with all entries 1 through n_arms
                active_list = list(range(n_arms))
                # Episode number for the pairwise elimination algorithm
                episode = 0
                # Initialize t = 1
                t = 1
                while t < horizon + 1:
                    # Get arm to be sampled per the PE-CS policy
                    k, omega_plus, active_list_plus, episode_plus = cs_pe(mu_hat, nsamps, horizon, last_sampled, omega,
                                                                          active_list, episode, alpha=subsidy_factor,
                                                                          kappa=0)
                    omega = omega_plus
                    active_list = active_list_plus
                    episode = episode_plus
                    if k is None:
                        continue
                    last_sampled = k
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
                    t += 1
            elif al == 'cs-etc':
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Array to hold how many times a certain arm is sampled
                nsamps = np.zeros(n_arms, dtype=np.int32)
                # Compute the value of tau based on the horizon and number of arms
                tau = 5 * int(np.ceil((horizon / n_arms) ** (2 / 3)))
                for t in range(1, horizon + 1):
                    # Get arm to be sampled per the ETC-CS policy
                    k = cs_etc(mu_hat, t, arm_cost_array, nsamps, horizon, tau, alpha=subsidy_factor)
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
            elif al == 'cs-ts':
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Array to hold how many times a certain arm is sampled
                nsamps = np.zeros(n_arms, dtype=np.int32)
                # Array to track number of successes and failures (TS Beta priors requires it)
                s_arms = np.zeros(n_arms, dtype=np.int32)
                f_arms = np.zeros(n_arms, dtype=np.int32)
                for t in range(1, horizon + 1):
                    # Get arm to be sampled per the CS-TS policy
                    k = cs_ts(mu_hat, s_arms, f_arms, arm_cost_array, t, alpha=subsidy_factor)
                    # Retrieve the pre-computed reward for this arm at this time-step
                    rew = arm_samples[k, t-1]
                    # Update s and f arrays manually here
                    s_arms[k] += rew
                    f_arms[k] += (1 - rew)
                    # Leave the rest to the book-keeping function
                    nsamps, mu_hat, qual_reg, cost_reg = do_bookkeeping_cost_subsidy(STEP, arm_samples, k, t, nsamps,
                                                                                     mu_hat, qual_reg, cost_reg, al, rs,
                                                                                     arm_reward_array, mu_calib,
                                                                                     arm_cost_array, c_calib)
            elif al == 'cs-ucb':
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Array to hold how many times a certain arm is sampled
                nsamps = np.zeros(n_arms, dtype=np.int32)
                for t in range(1, horizon + 1):
                    # Get arm to be sampled per the CS-TS policy
                    k = cs_ucb(mu_hat, arm_cost_array, t, nsamps, horizon, alpha=subsidy_factor)
                    # Leave the rest to the book-keeping function
                    nsamps, mu_hat, qual_reg, cost_reg = do_bookkeeping_cost_subsidy(STEP, arm_samples, k, t, nsamps,
                                                                                     mu_hat, qual_reg, cost_reg, al, rs,
                                                                                     arm_reward_array, mu_calib,
                                                                                     arm_cost_array, c_calib)
            else:
                print("Invalid algorithm {0} selected_algos, ignored".format(al))
                continue
