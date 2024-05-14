from math import log
import numpy as np
from numpy.random import beta
from src.policy_library.utils import *
from typing import Union


def bai(mu_hat: np.array, horizon: int, omega: np.array, B: list):
    """
    The bai algorithm based on improved UCB. Called only until > 1 arms in B
    :param mu_hat: Array (np.float) to hold the empirical return estimates mu_hat
    :param horizon: Known horizon budget as input
    :param omega: Array (int) to hold the terminal round numbers omega
    :param B: Set of active arms
    :return: Updated terminal round numbers omega, updated active arm set B
    """
    # Infer value of delta_tilde
    delta_tilde = pow(2.0, -np.max(omega))
    # Compute the value of tau (called n_m in the improved UCB paper)
    tau = int(np.ceil(2 * np.log(horizon * delta_tilde**2) / delta_tilde**2))
    # Compute the logarithmic buffer term
    beta_buffer = np.sqrt(np.log(horizon * delta_tilde**2) / (2 * tau))
    # Compute the UCB values for all arms
    ucb_values = mu_hat + beta_buffer
    # Compute the LCB values for all arms
    lcb_values = mu_hat - beta_buffer
    # Determine the highest LCB value in the set of active arms
    lcb_max = np.max(lcb_values[B])
    # Determine the updated set of active arms
    B_plus = [idx for idx in B if ucb_values[idx] > lcb_max]
    # Update the terminal round numbers for the arms in the updated active set
    omega_plus = np.copy(omega)
    omega_plus[B_plus] = omega[B_plus] + 1
    return omega_plus, B_plus


def sym_pe(mu_hat: np.array, ell: int, nsamps: np.array, horizon: int, omega: np.array, ep: Union[int, None],
           alpha: float):
    """
    :param mu_hat: Array (np.float) to hold the empirical return estimates mu_hat
    :param ell: Reference arm ell against which cheaper candidate arms are compared
    :param nsamps: Array (int) to hold the number of times each arm has been sampled
    :param horizon: Known horizon budget as input
    :param omega: Array (int) to hold the terminal round numbers omega
    :param ep: Counter for the episode that is currently active
    :param alpha: Subsidy factor to multiply the highest return by, lies in [0, 1]
    :return: Arm k to be sampled, updated terminal round numbers omega, updated episode number ep
    """
    # Get the gap delta_tilde as the gap associated with the current episode number
    delta_tilde = pow(2.0, -omega[ep])
    # Get the value of tau (n_m in the improved UCB paper)
    tau = int(np.ceil(2 * np.log(horizon * delta_tilde**2) / delta_tilde**2))
    for idx in [ep, ell]:
        if nsamps[idx] < tau:
            k = idx
            return k, omega, ep # Terminal round number and episode numbers remain unchanged
    # Compute the logarithmic buffer term
    beta_buffer = np.sqrt(np.log(horizon * delta_tilde**2) / (2 * tau))
    # Pre-compute the cost subsidy pre-multiplier
    alpha_prime = 1 - alpha
    # Check if `ell` should be eliminated in favor of arm `ep`
    if alpha_prime * (mu_hat[ell] + beta_buffer) < mu_hat[ep] - beta_buffer:
        k = ep
        ep_plus = None
        return k, omega, ep_plus
    # Else check if arm `ep` should be eliminated in favor of arm `ell`
    elif mu_hat[ep] + beta_buffer < alpha_prime * (mu_hat[ell] - beta_buffer):
        k = ep + 1
        ep_plus = ep + 1
        return k, omega, ep_plus
    else:
        k = ep
        ep_plus = ep
        omega_plus = np.copy(omega)
        omega_plus[ep] = omega[ep] + 1
        omega_plus[ell] = max(omega[ell], omega_plus[ep])
        return k, omega_plus, ep_plus


def pe(mu_hat: np.array, ell: int, nsamps: np.array, horizon: int, omega: np.array, ep: Union[int, None],
       alpha: float, kappa: int):
    """
    :param mu_hat: Array (np.float) to hold the empirical return estimates mu_hat
    :param ell: Reference arm ell against which cheaper candidate arms are compared
    :param nsamps: Array (int) to hold the number of times each arm has been sampled
    :param horizon: Known horizon budget as input
    :param omega: Array (int) to hold the terminal round numbers omega
    :param ep: Counter for the episode that is currently active
    :param alpha: Subsidy factor to multiply the highest return by, lies in [0, 1]
    :param kappa: Kappa is the maximum deviation between the round number for reference arm ell (leading) with the
    round number for the episode arm ep (trailing)
    :return:
    """
    # Init a two entry dict to hold the beta_buffer for both arms
    beta_buffer = {}
    delta_tilde = {}
    delta_tilde[ep] = pow(2.0, -omega[ep])
    delta_tilde[ell] = pow(2.0, -min(omega[ell], omega[ep] + kappa))
    # Get the gap delta_tilde as the gap associated with the current episode number
    for idx in [ep, ell]:
        # Retrieve the appt value of delta_tilde
        delta_tilde_idx = delta_tilde[idx]
        tau_idx = int(np.ceil(2 * np.log(horizon * delta_tilde_idx**2) / delta_tilde_idx**2))
        if nsamps[idx] < tau_idx:
            k = idx
            return k, omega, ep
        # Compute the logarithmic buffer term
        beta_buffer[idx] = np.sqrt(np.log(horizon * delta_tilde_idx**2) / (2 * tau_idx))
    alpha_prime = 1 - alpha
    # Check if `ell` should be eliminated in favor of arm `ep`
    if alpha_prime * (mu_hat[ell] + beta_buffer[ep]) < mu_hat[ep] - beta_buffer[ep]:
        k = ep
        ep_plus = None
        return k, omega, ep_plus
    # Else check if arm `ep` should be eliminated in favor of arm `ell`
    elif mu_hat[ep] + beta_buffer[ep] < alpha_prime * (mu_hat[ell] - beta_buffer[ell]):
        k = ep + 1
        ep_plus = ep + 1
        return k, omega, ep_plus
    else:
        k = ep
        ep_plus = ep
        omega_plus = np.copy(omega)
        omega_plus[ep] = omega[ep] + 1
        omega_plus[ell] = max(omega[ell], omega_plus[ep])
        return k, omega_plus, ep_plus


def cs_pe(mu_hat: np.array, nsamps: np.array, horizon: int, last_sampled: Union[int, None], omega: np.array,
          B: list, ep: Union[int, None], alpha: float=0.0, kappa: int=5):
    """
    Our two phase algorithm to compare against the CS-ETC algorithm
    After the first phase is done, the calling loop for this function resets the parameters
    so that they are ready for the next phase
    :param mu_hat: Array (np.float) to hold the empirical return estimates mu_hat
    :param nsamps: Array (int) to hold the number of times each arm has been sampled
    :param horizon: Known horizon budget as input
    :param last_sampled: Array index of the arm that was sampled last
    :param omega: Array (int) to hold the terminal round numbers omega
    :param B: List of active arms
    :param ep: Counter for the episode number being performed in phase 2
    :param alpha: Subsidy factor to multiply the highest return by, lies in [0, 1]
    :param kappa: Kappa is the maximum deviation between the round number for reference arm ell (leading) with the
     round number for the episode arm ep (trailing)
    :return: Arm k to be sampled, updated terminal round numbers omega, updated active arm set B,
     updated episode number ep
    """
    # Define variable to avoid editor warnings
    ell = B[0]  # ell is only used when there is only a single active arm left
    # Check if arms are yet to be eliminated
    if len(B) > 1:
        delta_tilde = pow(2.0, -np.max(omega))
        tau = int(np.ceil(2 * np.log(horizon * delta_tilde**2) / delta_tilde**2))
        for idx in B:
            if nsamps[idx] < tau:
                k = idx
                return k, omega, B, ep
        omega_plus, B_plus = bai(mu_hat, horizon, omega, B)
        if len(B_plus) == 1:
            # No arm to be sampled, so k is set to None
            k = None
            omega_plus = omega  # Go back to old omega
        else:
            k = B_plus[0]
        return k, omega_plus, B_plus, ep
    elif ep not in [None, ell]:
        k, omega_plus, ep_plus = pe(mu_hat, ell, nsamps, horizon, omega, ep, alpha, kappa)
        return k, omega_plus, B, ep_plus
    else:
        return last_sampled, omega, B, ep   # Have reached the end of search, just need to sample to exhaust budget


def cs_ucb(mu_hat: np.array, costs: np.array, t: int, nsamps, horizon: int, alpha: float=0.0):
    """
    Implementation of the CS-UCB algorithm as described in the MAB-CS paper
    https://proceedings.mlr.press/v130/sinha21a.html
    ***********************
    :param mu_hat: Array of empirical mean rewards
    :param costs: Array of known fixed costs
    :param t: Time step
    :param nsamps: Array of number of samples associated with each arm
    :param horizon: Int for horizon budget
    :param alpha: Subsidy factor, lies in [0, 1]
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
        ell_t = random_argmax(I_ucb)
        # Receive all the indices of feasible arms
        feasible_arms = np.where(I_ucb >= (1 - alpha) * I_ucb[ell_t])[0]
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


def cs_ts(mu_hat: np.array, s_arms: np.array, f_arms: np.array, costs: np.array, t: float, alpha: float=0.0):
    """
    Implementation of the CS-TS algorithm as described in the MAB-CS paper
    https://proceedings.mlr.press/v130/sinha21a.html
    Thompson sampling with uniform prior and beta likelihood
    In addition to the book-keeping of a regular bandit algorithm, also track
    successes (s_arms) and failures (f_arms)
    ***********************
    :param mu_hat: Array of empirical mean rewards
    :param s_arms: Number of successes
    :param f_arms: Number of failures
    :param costs: Array of known fixed costs
    :param t: Time step
    :param alpha: Subsidy factor, lies in [0, 1]
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
        feasible_arms = np.where(samples >= (1 - alpha) * samples[m_t])[0]
        # Determine the cost of the cheapest arm in the set of feasible arms
        min_cost = np.min(costs[feasible_arms])
        # Get all the feasible indices that can match this cost
        min_cost_indices_overall = np.where(costs == min_cost)[0]
        # Take the intersection of feasible arms and minimum cost indices overall
        feasible_min_cost_indices = np.intersect1d(feasible_arms, min_cost_indices_overall)
        # Select a random index among these min cost indices
        k = np.random.choice(feasible_min_cost_indices)
    return k


def cs_etc(mu_hat: np.array, t: float, costs: np.array, nsamps: np.array, horizon: int, tau: int, alpha=0.0):
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
    :param tau: Number of times each arm is pre-explored in phase 1 of the algorithm
    :param alpha: Subsidy factor to multiply the highest return by, lies in [0, 1]
    :return
    """
    # Infer the number of arms
    n_arms = len(mu_hat)
    # Phase 1: Pure exploration
    if t < n_arms * tau:
        # Sample the arm with index (t % n_arms)
        k = t % n_arms
        return k
    # Move onto the UCB phase of the algorithm
    else:
        # Compute an array of UCB buffer terms
        buffer = np.sqrt(2 * log(horizon) / nsamps)
        # Compute the UCB values for all arms
        ucb_values = np.minimum(mu_hat + buffer, 1.0)
        # Compute the LCB values for all the arms
        lcb_values = np.maximum(mu_hat - buffer, 0.0)
        # Compute m_t for constructing feasible set as the arm with the highest LCB
        ell_t = np.argmax(lcb_values)
        # Construct the feasible set as the arm-indices of arms having
        #  UCB above the subsidized LCB of arm m_t
        feasible_arms = np.where(ucb_values >= (1 - alpha) * lcb_values[ell_t])[0]
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
