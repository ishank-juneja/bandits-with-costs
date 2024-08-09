# Fork of simulate policies FCS to run only TS-CS
#  where we specify the random seeds to run on
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
args = parser.parse_args()
# Get the input bandit instance file_name
in_file = args.file

# Min horizon for the simulation
MIN_HORIZON = int(5e6)

# Horizon/ max number of iterations
horizon = int(args.horizon)
assert horizon >= MIN_HORIZON, "Horizon too small, must be at least 5e6"
# Step interval for which data is recorded
STEP = args.STEP

# selected_seeds = [38]
# selected_seeds = [38, 87, 93]
selected_seeds = np.arange(91, 101, 1)


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
    al = 'cs-ts'
    for rs in selected_seeds:
        # Set numpy random seed to make output deterministic for a given run
        np.random.seed(rs)
        # Simulate/Pre-compute all the rewards we will need in this run
        arm_samples = simulate_bandit_rewards(arm_reward_array, MIN_HORIZON)
        # Initialize expected cumulative cost and quality regret
        qual_reg = 0.0
        cost_reg = 0.0
        # For every run of every algorithm, prepend the (t = 0, qual_reg = 0, cost_reg = 0, nsamps = 0) data point
        #  to the output file
        sys.stdout.write("{0}, {1}, {2}, {3:.2f}, {4:.2f}, {5}\n".format(al, rs, 0, qual_reg, cost_reg,
                                                                         ';'.join(['0'] * n_arms)))
        if al == 'cs-ts':
            # Array to hold empirical estimates of each arms reward expectation
            mu_hat = np.zeros(n_arms)
            # Array to hold how many times a certain arm is sampled
            nsamps = np.zeros(n_arms, dtype=np.int32)
            # Array to track number of successes and failures (TS Beta priors requires it)
            s_arms = np.zeros(n_arms, dtype=np.int32)
            f_arms = np.zeros(n_arms, dtype=np.int32)
            for t in range(1, horizon + 1):
                # Collect new arm samples every MIN_HORIZON time-steps
                if (t - 1) % MIN_HORIZON == 0 and t != 1:
                    arm_samples = simulate_bandit_rewards(arm_reward_array, MIN_HORIZON)
                # Get arm to be sampled per the CS-TS policy
                k = cs_ts(mu_hat, s_arms, f_arms, arm_cost_array, t, alpha=subsidy_factor)
                # Retrieve the pre-computed reward for this arm at this time-step
                rew = arm_samples[k, (t-1) % MIN_HORIZON]
                # Update s and f arrays manually here
                s_arms[k] += rew
                f_arms[k] += (1 - rew)
                # Increment number of times kth arm sampled
                nsamps[k] = nsamps[k] + 1
                # Update empirical reward estimates, compute new empirical mean
                mu_hat[k] = ((nsamps[k] - 1) * mu_hat[k] + rew) / nsamps[k]
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
        else:
            print("Invalid algorithm {0} selected_algos, ignored".format(al))
            continue
