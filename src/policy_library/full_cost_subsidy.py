import numpy as np
from src.policy_library.no_cost_subsidy import improved_ucb
from src.policy_library.reference_ell_setting import pairwise_elimination


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
