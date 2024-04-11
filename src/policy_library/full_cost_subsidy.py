from math import log
import numpy as np
from numpy.random import beta
from src.policy_library.no_cost_subsidy import improved_ucb
from src.policy_library.reference_ell_setting import pairwise_elimination_for_cs_pe
from src.policy_library.utils import *


# Set a more apt name for the improved_ucb function in the cs-pe context
bai = improved_ucb


def cs_pe(mu_hat: np.array, nsamps: np.array, horizon: int, last_sampled: int, delta_tilde: float, B: list,
          episode: int, omega: np.array, alpha: float=0.0):
    """
    Our two phase algorithm to compare against the CS-ETC algorithm
    After the first phase is done, the calling loop for this function resets the parameters
    so that they are ready for the next phase
    :param mu_hat: Array (np.float) to hold the empirical return estimates mu_hat
    :param nsamps: Array (int) to hold the number of times each arm has been sampled
    :param horizon: Known horizon budget as input
    :param last_sampled: Index of the last arm sampled to check if already sampled sufficiently
    :param delta_tilde: The iterative gap used by the algorithm
    :param B: List of active arms
    :param episode: Counter to track the phase of the PE algorithm
    :param omega: Array (float) to hold the iterative gap value at which every arm is eliminated in phase 1
        used to find where to pick back up from in phase 2
    :param alpha: Subsidy factor to multiply the highest return by, lies in [0, 1]
    :return:
    """
    # Check if there are still more than 1 arms left, if so, run phase 1 again
    if len(B) > 1:
        # Run phase 1 Improved-UCB elimination phase code
        k, delta_tilde_new, B_new, omega = bai(mu_hat, nsamps, horizon, delta_tilde, B, last_sampled, omega)
    else:
        # There is no change to the set of active arms (in fact we are in phase 2, and notion of active arms is gone)
        B_new = B
    # Actually sample the prescription made by phase 1, only if there remain to be more than 1 arms in B_new
    if len(B_new) > 1:
        return k, delta_tilde_new, B_new, episode # Only entered when k has been set by the return value of bai
    # Else pick the best arm declared by imp UCB as reference and move onto phase 2
    else:
        # Infer the reference arm as the only arm in B_new
        ref_arm = B_new[0]
        # Run phase 2 PE code
        k, delta_tilde_new, episode_new = pairwise_elimination_for_cs_pe(ref_arm, mu_hat, nsamps, horizon, delta_tilde,
                                                                         episode, last_sampled, omega, alpha)
        return k, delta_tilde_new, B_new, episode_new


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
        I_ucb_raw = mu_hat + np.sqrt(2 * log(t) / nsamps)
        I_ucb = np.minimum(I_ucb_raw, 1.0)
        # Identify the arm with the highest index and treat it as the proxy for the
        #  best arm
        m_t = random_argmax(I_ucb)
        # Receive all the indices of feasible arms
        feasible_arms = np.where(I_ucb > (1 - alpha) * I_ucb[m_t])[0]
        # Determine the value of the cheapest cost in the feasible set
        min_cost = np.min(costs[feasible_arms])
        # Get all the indices in original array where the cost
        #  equals min_cost
        min_cost_indices_overall = np.where(costs == min_cost)[0]
        # Take the intersection of feasible arms and minimum cost indices overall
        feasible_min_cost_indices = np.intersect1d(feasible_arms, min_cost_indices_overall)
        # Select a random index among these min cost indices
        k = np.random.choice(feasible_min_cost_indices)
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
    samples_raw = np.zeros_like(s_arms, dtype=np.float32)
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
        min_cost = np.min(costs[feasible_arms])
        # Get all the feasible indices that can match this cost
        min_cost_indices_overall = np.where(costs == min_cost)[0]
        # Take the intersection of feasible arms and minimum cost indices overall
        feasible_min_cost_indices = np.intersect1d(feasible_arms, min_cost_indices_overall)
        # Select a random index among these min cost indices
        k = np.random.choice(feasible_min_cost_indices)
    return k


def cs_etc(mu_hat, t, costs, nsamps, horizon, last_sampled, tau, alpha=0.0):
    """
    Implementation of the CS-ETC algorithm from the MAB-CS paper
    https://proceedings.mlr.press/v130/sinha21a.html
    ***********************
    Algo explores each arm for a fixed pre-computed budget
     and then exploits per its own exploitation rules
    ***********************
    :param mu_hat: Array to hold the empirical return estimates mu_hat
    :param t: Time step
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
        buffer = np.sqrt(2 * log(t) / nsamps)
        # Compute the UCB values for all arms
        ucb_values = np.minimum(mu_hat + buffer, 1.0)
        # Compute the LCB values for all the arms
        lcb_values = np.maximum(mu_hat - buffer, 0.0)
        # Compute m_t for constructing feasible set as the arm with the
        #  highest LCB
        m_t = np.argmax(lcb_values)
        # Construct the feasible set as the arm-indices of arms having
        #  UCB above the subsidized LCB of arm m_t
        feasible_arms = np.where(ucb_values > (1 - alpha) * lcb_values[m_t])[0]
        # Get the minimum cost among feasible arms
        min_cost = np.min(costs[feasible_arms])
        # Get all the indices of the original array where the cost of the arm is equal to the
        #  minimum cost
        min_cost_indices_overall = np.where(costs == min_cost)[0]
        # Take intersection of feasible arms and minimum cost indices overall
        feasible_min_cost_indices = np.intersect1d(feasible_arms, min_cost_indices_overall)
        # Select a random index among these min cost indices
        k = np.random.choice(feasible_min_cost_indices)
        return k
