import numpy as np
from math import ceil, log, sqrt


def pairwise_elimination(mu_hat, nsamps, horizon, delta_tilde, episode_num, last_sampled):
    """
    A function that implements the Successive Pairwise Elimination algorithm from the notes
    https://www.overleaf.com/project/6502fd4306f4b073aa6bd809
    ***********************
    Through every execution of this function we sample a single arm
    The round number m is not tracked explicitly, but instead it is tracked
     implicitly using the algorithm's proxy gap tilde{Delta}
    Case 1: All the episodes 1 through ell - 1 have been completed
    if so, we simply sample the arm ell until the horizon budget is exhausted
    Case 2: An episode j for evaluating the candidacy of arm with index j is ongoing
    if so, we check if the last sampled arm has been sampled n_m times already
    if not, sample that arm again,
    if yes, sample the other of the two active arms until it has been sampled at least n_m times
    if both arms have been sampled at least n_m times, we move to the arm elimination phase
    Comparisons in lines 14 and 18 of the algo block happen ...
    If the current call of the function was at the cusp of a new episode (candidate arm elimination occurs),
     then in the call we will perform the comparison and if,
     (i) Arm j gets eliminated by arm ell, then we set next arm to be sampled to be j + 1 (j + 1 could equal ell
     which is fine since if we have reached the end of the pack, we sample ell for the entire budget anyway)
     (ii) Arm ell gets eliminated by arm j, then we set next arm to be sampled to be j and set the proxy gap
     delta_tilde to be -1 to indicate that the least cost acceptable arm has been identified.
     It is the job of the calling function to recognize this and set the episode number to be -1
     so that the next iteration onwards the comparisons are not performed
    ***********************
    mu_hat_ell: Empirical estimates of rewards for each candidate arm and arm ell
     Therefore the length of mu_hat is ell
    nsamps: Number of times each arm has been sampled
    horizon: Known horizon as input
    delta_tilde: Gaps used by the elimination to set number of samples in a batch and
     UCB buffer terms, reset for both arm ell and candidate arm j at the start of
     every new episode
    :param episode_num: Index of the current episode, varies between 0 and ell - 1
    :param last_sampled: The arm that was being sampled in the previous call to the function since
     the same arm will be sampled until we accumulate n_m samples for it
    :return: Index of the arm to be sampled, updated delta_tilde
    """
    # Infer the maximum number of episodes from the length of mu_hat
    ell = len(mu_hat)
    # If episode ell has been hit in the simulation keep returning ell
    if episode_num == ell:
        return ell, delta_tilde
    # If episode ell has not been hit, then check if the least cost acceptable arm has been identified
    #  as indicated via an invalid episode number of -1
    elif episode_num == -1:
        # In this case the last sampled arm is the arm j that is the successful least cost candidate
        #  and we keep returning it until the horizon budget is exhausted
        return last_sampled, delta_tilde
    # Else if the episode number is a regular valid episode number then assume that the
    #  episode in question is on going
    else:
        # Recompute n_m for the current episode
        n_m = ceil(2 * log(horizon * delta_tilde ** 2) / (delta_tilde ** 2))
        # Check the number of times the episode_num candidate arm has been sampled
        if nsamps[episode_num] < n_m:
            # If it has not been sampled n_m times, return it again with
            #  all other parameters unchanged
            return episode_num, delta_tilde
        # Else arm j (episode_num) must have been sampled n_m times already, so check samples of ell
        elif nsamps[ell] < n_m:
            # If ell has not been sampled n_m times, return it with all other parameters unchanged
            return ell, delta_tilde
        # Else both arms have been sampled at least n_m times, so move to the arm elimination phase
        else:
            # Compute the buffer terms for UCB/LCB
            buffer = sqrt(log(horizon * delta_tilde ** 2) / (2 * n_m))
            # Check if arm ell should be eliminated in favor of arm j and the episodes concluded
            if mu_hat[ell] + buffer < mu_hat[episode_num] - buffer:
                # Set delta_ number to be -1 to indicate that the least cost acceptable
                # arm has been identified, and return arm j for sampling
                delta_tilde = -1.0 # From this caller infers that the comparisons are over
                k = episode_num
                return k, delta_tilde
            elif mu_hat[episode_num] + buffer < mu_hat[ell] - buffer:
                # Go to next episode
                k = episode_num + 1
                # Reset delta_tilde for the next episode
                delta_tilde = 1.0
                return k, delta_tilde
            else:
                # Continue with the next round of the same episode by updating delta_tilde
                delta_tilde = delta_tilde / 2
                # Set the next arm to be sampled to be arm j since we will certainly
                #  need more samples from it whereas, we might not need more samples from ell immediately
                k = episode_num
                return episode_num, delta_tilde


def improved_ucb(mu_hat, nsamps, horizon, delta_tilde, B, last_sampled):
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
     we will perform the elimination and then set arm k to be
     the smallest index arm in the active set
    ***********************
    mu_hat: Empirical estimates of rewards for each arm
    nsamps: Number of times each arm has been sampled
    horizon: Known horizon as input
    delta_tilde: Gaps used by the elimination to set number of samples in a batch and
     UCB buffer terms
    B: List of arms that are not yet eliminated
    last_sampled: Index of the last sampled arm so that we know which arm to sampled next
     in batched/rounded sampling
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

    for arm_indices_k in B:
        # Compute the UCB of arm k
        ucb_k = mu_hat[arm_indices_k] + buffer
        # Check if the UCB of arm k is under the LCB of any other arm
        if ucb_k >= max(active_lcb_list):
            # If not, add arm k to the new list
            B_new.append(arm_indices_k)

    # Replace the old list with the new list
    B = B_new

    # Update delta_tilde
    delta_tilde = delta_tilde / 2
    # Return package says sample the lowest index arm in the set of active arms
    k = B[0]
    # Return all other parameters
    return k, delta_tilde, B


def random_argmin(arr):
    """
    A wrapper around np.argmin that breaks ties between minimum index options uniformly at random.
    """
    # Find the minimum value in the array
    min_val = np.min(arr)

    # Find all indices where this minimum value occurs
    min_indices = np.where(arr == min_val)[0]

    # Select one of these indices at random and return
    return np.random.choice(min_indices)


# pestimates are empirical estimate of probabilities
# nsamps is number of times each arm is sampled
def UCB(p_estimates, nsamps, t):
    # Update ucb value
    I_ucb = p_estimates + np.sqrt(2 * log(t) / nsamps)
    # Determine arm to be sampled in current step,
    # Ties are broken by lower index preference
    k = np.argmax(I_ucb)
    return k


def MTR_UCB(p_estimates, nsamps, t, costs, theta):
    # Compute the upper confidence bounds for all arms
    I_ucb = p_estimates + np.sqrt(2 * np.log(t) / nsamps)

    # Filter arms with UCB value greater than theta
    valid_arms = np.where(I_ucb > theta)[0]

    # If no arm has UCB greater than theta, return a random arm
    if len(valid_arms) == 0:
        return np.random.choice(range(len(p_estimates)))

    # Among the valid arms, find the one with the smallest cost
    min_cost = np.min(costs[valid_arms])
    # Get all indices from the original array where the cost equals the minimum cost
    min_cost_indices = np.where(costs == min_cost)[0]

    # Select a random index from these minimum cost indices
    k = np.random.choice(min_cost_indices)

    return k


if __name__ == '__main__':
    # Example usage
    arr = np.array([1, 0, 0, 2, 0])
    random_min_index = random_argmin(arr)
    print(random_min_index)

    # Test function

    # Create a scenario for testing
    np.random.seed(4)  # Seed for reproducibility
    p_estimates = 0.1 * np.random.rand(5)  # Random success probabilities for 5 arms
    nsamps = np.array([10, 20, 15, 25, 30])  # Number of times each arm has been pulled
    t = 50  # Current time step
    costs = np.array([3, 1, 2, 5, 0.5])  # Costs for pulling each arm
    theta = 0.7  # Threshold value for UCB

    # Run the test
    selected_arm = MTR_UCB(p_estimates, nsamps, t, costs, theta)

    print("Selected arm:", selected_arm)
    print("p_estimates:", p_estimates)
    print("nsamps:", nsamps)
    print("costs:", costs)
    print("UCB values:", p_estimates + np.sqrt(2 * np.log(t) / nsamps))
