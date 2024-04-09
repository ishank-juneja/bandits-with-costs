import numpy as np
from math import ceil, log, sqrt


# pestimates are empirical estimate of probabilities
# nsamps is number of times each arm is sampled
def UCB(p_estimates, nsamps, t):
    # Update ucb value
    I_ucb = p_estimates + np.sqrt(2 * log(t) / nsamps)
    # Determine arm to be sampled in current step,
    # Ties are broken by lower index preference
    k = np.argmax(I_ucb)
    return k


def improved_ucb(mu_hat, nsamps, horizon, delta_tilde, B, last_sampled, omega: np.array=None):
    """
    A function that implements the Improved UCB algorithm.
    https://www.overleaf.com/project/6502fd4306f4b073aa6bd809
    ***********************
    Through every execution of this function we sample a single arm
    The round number m is not tracked explicitly, but instead it is tracked
     implicitly using the algorithm's proxy gap tilde{Delta}
    Case 1: A single active arm is left in B_m
    if so then we must sample that arm until horizon is exhausted
    Case 2: Multiple arms are left in B_m and a round is ongoing
    if so we check if the most recently sampled arm has been sampled n_m times
    if not, we sample that arm again
    if yes, we move to the arm with the next highest index in B_m if there is one
    if there is no next arm in B_m, we end the round, and move to arm elimination ...
    While eliminating arms, we use the criteria for eliminating an arm as given in the paper
    and update the round number m, delta_tilde, and B_m accordingly
    If the current call of the function was at the cusp of a new round then in the call
     we will perform the elimination and then set arm k to be the smallest index arm in the
     active set (since we are going to need more samples from it)
    ***********************
    mu_hat: Empirical estimates of rewards for each arm
    nsamps: Number of times each arm has been sampled
    horizon: Known horizon as input
    delta_tilde: Gaps used by the elimination to set number of samples in a batch and
     UCB buffer terms
    B: List of arms that are not yet eliminated
    last_sampled: Index of the last sampled arm so that we know which arm to sampled next
     in batched/rounded sampling
     omega: An array to hold the final round to which an arm being eliminated survived
    :return: Index of the arm to be sampled, updated delta_tilde, and updated B
    """
    # Check if there is only one arm left in B_m
    if len(B) == 1:
        # If so, keep returning that arm until the calling horizon for loop gets terminated
        k = B[0]
        # Returned parameters other than k are effectively don't cares
        return k, delta_tilde, B
    # Else if there is more than one arm and an ongoing round m, then keep
    #  batch sampling arms until n_m samples have been collected for each of them
    elif len(B) > 1:
        # Recompute n_m for the current round m
        n_m = ceil(2 * log(horizon * delta_tilde ** 2) / (delta_tilde ** 2) )
        # Check the number of times the last_sampled arm has been sampled
        if nsamps[last_sampled] < n_m:
            # If it has not been sampled n_m times, return it again with
            #  all other parameters unchanged
            return last_sampled, delta_tilde, B
        elif nsamps[last_sampled] == n_m:
            # If sampled n_m times, move to the next arm in B_m
            #  If there is no next arm, end the round
            if last_sampled == B[-1]:
                # Round is complete, move to arm elimination phase
                pass
            else:
                # Else, move to the next arm in B_m
                k = B[B.index(last_sampled) + 1]
                # Return all other parameters unchanged
                return k, delta_tilde, B
    # Start arm elimination phase
    # Compute the buffer terms for UCB/LCB indices
    buffer = sqrt( log(horizon * delta_tilde ** 2) / (2 * n_m) )
    # Using mean and buffer, compute the LCB of all the active arms
    active_lcb_list = [mu_hat[k] - buffer for k in B]

    # Initialize a new list to hold the indices of the arms to keep
    B_new = []

    # Iterate over the surviving indices
    for arm_indices_k in B:
        # Compute the UCB of arm k
        ucb_k = mu_hat[arm_indices_k] + buffer
        # Eliminate arms whose UCB has fallen below the largest LCB in the set of active arms by
        #  keeping only the arms whose UCB is greater than the largest LCB
        if ucb_k >= max(active_lcb_list):
            # If not, keep arm k in the new list
            B_new.append(arm_indices_k)
        elif omega is not None:
            # if an arm is being eliminated and if omega is a parameter that was passed
            #  then set the round in which the arm was eliminated
            omega[arm_indices_k] = delta_tilde

    # Replace the old list with the new list
    B = B_new

    # Update delta_tilde
    delta_tilde = delta_tilde / 2
    # Return package says sample the lowest index arm in the set of active arms
    k = B[0]
    # Return all other parameters
    return k, delta_tilde, B
