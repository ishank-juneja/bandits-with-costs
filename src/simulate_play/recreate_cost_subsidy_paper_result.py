# Single script to recreate the results of the cost subsidy paper
import numpy as np
from src.utils import do_bookkeeping_cost_subsidy, simulate_bandit_rewards
import sys
from src.instance_handling.get_instance import read_instance_from_file
from src.policy_library import cs_etc, cs_pe


# Command line inputs
algos = ['etc-cs']
# Horizon/ max number of iterations
horizon = 5000
# Number of runs to average over
nruns = 50
# Step interval for which data is recorded
STEP = 1


if __name__ == '__main__':


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
            sys.stdout.write("{0}, {1}, {2}, {3:.2f}, {4:.2f}, {5}\n".format(al, rs, 0, qual_reg, cost_reg,
                                                                             ';'.join(['0'] * n_arms)))
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
                    k = cs_etc(mu_hat, arm_cost_array, nsamps, horizon, last_sampled, tau, alpha=subsidy_factor)
                    # Update last_sampled
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
            else:
                print("Invalid algorithm {0} selected_algos, ignored".format(al))
                continue
