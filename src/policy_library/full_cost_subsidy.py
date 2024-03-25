import numpy as np


def CS_ETC(mu_hat, costs, nsamps, horizon, last_sampled, tau, alpha=0.0):
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
        ucb_values = np.max(mu_hat + buffer, np.ones_like(mu_hat))
        # Compute the LCB values for all the arms
        lcb_values = np.max(mu_hat - buffer, np.zeros_like(mu_hat))
        # Compute m_t for constructing feasible set as the arm with the
        #  highest LCB
        m_t = np.argmax(lcb_values)
        # Construct the feasiable set as the arm-indices of arms having
        #  UCB above the subsidized LCB of arm m_t
        feasible_set = np.where(ucb_values > (1 - alpha) * lcb_values[m_t])[0]
        # Return the least cost arm in the feasible set to be sampled
        return np.argmin(costs[feasible_set])


# This function is going to be the phase 1 BAI step of my
#  2 phase solution to the cost subsidy problem
def phase1_improved_ucb()