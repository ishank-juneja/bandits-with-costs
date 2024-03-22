import numpy as np
from math import log, sqrt
import math


def conservative_UCB_known_mu_o(p_estimates, nsamps, t, delta, mu_o, alpha):
    """
    Implementation of the conservative UCB algorithm:
    https://proceedings.mlr.press/v48/wu16.pdf#:~:text=Conservative%20UCB%20(Algorithm%201)
    This algorithm assumes that the first arm (index 0) is the "default arm" and
    :param p_estimates: array to hold mu_hat
    :param nsamps: Array to hold number of times each arm has been sampled
    :param t:
    :param delta: The desired confidence level on the constraint being satisfied
    :param mu_o: The reward threshold to exceed
    :param alpha: the subsidy factor
    :return: Index of the arm to be sampled
    """
    # Infer the number of arms from the length of the p_estimates array
    n_arms = len(p_estimates)

    # Function to compute the quantity needed
    def psi_delta_of_s(s: int, delta: float, narms: np.array):
        # Compute ratio zeta
        zeta = narms / delta
        ret = (log(max(3., log(zeta))) + log(2 * math.e**2 * zeta) + (zeta * (1 + log(zeta))) / ((zeta - 1) * log(zeta)) * log(log(1. + s)))
        return ret

    # Initialize arrays to hold the confidence intervals for all the arms
    theta_arr = np.zeros_like(p_estimates)  # Upper bounds
    lambda_arr = np.zeros_like(p_estimates) # Lower bounds
    # Freeze upper and lower bounds for the default arm
    theta_arr[0] = mu_o
    lambda_arr[0] = mu_o
    # Compute the confidence intervals for the other arms
    for idx in range(1, n_arms):
        del_i = sqrt(psi_delta_of_s(nsamps[idx], delta, n_arms) / nsamps[idx])
        theta_arr[idx] = p_estimates[idx] + del_i
        lambda_arr[idx] = p_estimates[idx] - del_i

    # Determine the arm to be sampled per UCB scheme
    # Ties are broken by lower index preference
    Jt = np.argmax(theta_arr)

    # Compute the budget
    xi = np.sum(lambda_arr * nsamps) + lambda_arr[Jt] - (1 - alpha) * t * mu_o

    # Make the final decision based on the slack in the budget
    if xi >= 0:
        k = Jt
    else:
        k = 0
    return k
