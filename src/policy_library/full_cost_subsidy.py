from math import log
import numpy as np
from numpy.random import beta
from src.policy_library.no_cost_subsidy import improved_ucb
from src.policy_library.reference_ell_setting import pairwise_elimination
from src.policy_library.utils import *


def cs_ucb(mu_hat, costs, t, nsamps, horizon, alpha=0.0):
    """
    Implementation of the CS-UCB algorithm as described in the MAB-CS paper
    https://proceedings.mlr.press/v130/sinha21a.html
    ***********************
    :param mu_hat:
    :param costs:
    :param nsamps:
    :param horizon:
    :param alpha:
    :return: The arm k to be sampled
    """
    # Infer the number of arms
    n_arms = len(mu_hat)
    # Initially sample all arms at-least once, then jump into the
    #  the cost-subsidy style decision rules
    if t < n_arms + 1:
        # Sample the arm with (array) index (t - 1)
        k = t - 1
    else:
        # Compute the UCB index associated with every single arm
        I_ucb_raw = mu_hat + np.sqrt(2 * log(horizon) / nsamps)
        I_ucb = np.minimum(I_ucb_raw, 1.0)
        # Identify the arm with the highest index and treat it as the proxy for the
        #  best arm
        m_t = random_argmax(I_ucb)
        # Receive all the indices of feasible arms
        feasible_arms = np.where(I_ucb > (1 - alpha) * I_ucb[m_t])[0]
        # Determine the value of the cheapest cost in the feasible set
        min_cost = np.min(feasible_arms)
        # Get all the indices in original array where the cost
        #  equals min_cost
        min_cost_indices = np.where(costs == min_cost)[0]
        # Select a random index among these min cost indices
        k = np.random.choice(min_cost_indices)
    return k


def cs_ts(mu_hat, s_arms, f_arms, costs, t, alpha=0.0):
    """
    Implementation of the CS-TS algorithm as described in the MAB-CS paper
    https://proceedings.mlr.press/v130/sinha21a.html
    Thompson sampling with uniform prior and beta likelihood
    In addition to the book-keeping of a regular bandit algorithm, also track
    successes (s_arms) and failures (f_arms)
    ***********************
    :param mu_hat:
    :param s_arms: Number of successes
    :param f_arms: Number of failures
    :param costs:
    :param t:
    :param alpha:
    :return: The arm k to be sampled
    """
    # Infer the number of arms
    n_arms = len(mu_hat)
    # Array to hold the observed samples_raw
    samples_raw = np.zeros_like(s_arms)
    # Initially sample all arms at-least once, then jump into the
    #  the cost-subsidy style decision rules
    if t < n_arms + 1:
        # Sample the arm with (array) index (t - 1)
        k = t - 1
    else:
        for idx in range(n_arms):
            # Create and sample a beta random variable for current arm
            samples_raw[idx] = beta(s_arms[idx] + 1, f_arms[idx] + 1)
        # Filter the samples_raw to max out at 1.0
        samples = np.minimum(samples_raw, 1.0)
        # Retrieve the index of the largest sample, breaking ties at random
        m_t = random_argmax(samples_raw)
        # Receive all the indices of the feasible arms
        feasible_arms = np.where(samples > (1 - alpha) * samples[m_t])[0]
        # Determine the cost of the cheapest arm in the set of feasible arms
        min_cost = np.min(feasible_arms)
        # Get all the feasible indices that can match this cost
        min_cost_indices = np.where(costs == min_cost)[0]
        # Select a random index among these min cost indices
        k = np.random.choice(min_cost_indices)
    return k


def cs_etc(mu_hat, costs, nsamps, horizon, last_sampled, tau, alpha=0.0):
    """
    Implementation of the CS-ETC algorithm from the MAB-CS paper
    https://proceedings.mlr.press/v130/sinha21a.html
    ***********************
    Algo explores each arm for a fixed pre-computed budget
     and then exploits per its own exploitation rules
    ***********************
    :param mu_hat: Array to hold the empirical return estimates mu_hat
    :param costs: Array to hold the costs of sampling each arm
    :param nsamps: Array to hold the number of times each arm has been sampled
    :param horizon: Known horizon budget as input
    :param last_sampled: Index of the last arm sampled to check if
     already sampled sufficiently
    :param tau: Number of times each arm is pre-explored in phase 1 of the algorithm
    :param alpha: Subsidy factor to multiply the highest return by, lies in [0, 1]
    :return
    """
    # Phase 1: Pure exploration
    # Check if the number of samples of the last sampled arm has hit tau
    if nsamps[last_sampled] < tau:
        # If it hasn't, then keep sampling it
        return last_sampled
    # Check if any arm is left to be sampled tau times
    elif np.min(nsamps) < tau:
        # If yes then move onto the next arm in the sequence
        #  and sample it
        return last_sampled + 1
    # Move onto the UCB phase of the algorithm
    else:
        # Compute an array of UCB buffer terms
        buffer = np.sqrt(2 * np.log(horizon) / nsamps)
        # Compute the UCB values for all arms
        ucb_values = np.minimum(mu_hat + buffer, 1.0)
        # Compute the LCB values for all the arms
        lcb_values = np.maximum(mu_hat - buffer, 0.0)
        # Compute m_t for constructing feasible set as the arm with the
        #  highest LCB
        m_t = np.argmax(lcb_values)
        # Construct the feasible set as the arm-indices of arms having
        #  UCB above the subsidized LCB of arm m_t
        feasible_set = np.where(ucb_values > (1 - alpha) * lcb_values[m_t])[0]
        # Return the least cost arm in the feasible set to be sampled
        return np.argmin(costs[feasible_set])


def cs_pe(mu_hat, nsamps, horizon, last_sampled, delta_tilde, B, episode, reference_arm, alpha=0.0):
    """
    Our two phase algorithm to compare against the CS-ETC algorithm
    After the first phase is done, the calling loop for this function resets the parameters
    so that they are ready for the next phase
    param mu_hat: Array to hold the empirical return estimates mu_hat
    :param costs: Array to hold the costs of sampling each arm
    :param nsamps: Array to hold the number of times each arm has been sampled
    :param horizon: Known horizon budget as input
    :param last_sampled: Index of the last arm sampled to check if
     already sampled sufficiently
    :param delta_tilde: Parameter to track the phase change in the PE algorithm
    :param B: List of arms that are still in contention
    :param episode: Counter to track the phase of the PE algorithm
    :param reference_arm: Arm declared as the best arm by the imp-UCB phase
    :param alpha: Subsidy factor to multiply the highest return by, lies in [0, 1]
    :return:
    """
    # Run phase 1 Improved-UCB elimination phase code
    k, delta_tilde, B = improved_ucb(mu_hat, nsamps, horizon, delta_tilde, B, last_sampled)
    # Check if there is > 1 arm, if so continue using phase 1
    if len(B) != 1:
        return k, delta_tilde, B, episode
    # Else pick the best arm declared by imp UCB as reference and move onto phase 2
    else:
        # Run phase 2 PE code
        k, delta_tilde, episode = pairwise_elimination(reference_arm, mu_hat, nsamps, horizon, delta_tilde, episode,
                                                       last_sampled, alpha)
        return k, delta_tilde, B, episode
