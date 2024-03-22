import numpy as np
import math
from math import ceil, log, sqrt


def ref_arm_ell_UCB(p_estimates, nsamps, t, costs, ref_arm_idx):
    # Compute the upper confidence bounds for all arms
    I_ucb = p_estimates + np.sqrt(2 * np.log(t) / nsamps)
    # Compute the LCB for the reference arm
    lcb_ell = p_estimates[ref_arm_idx] - np.sqrt(2 * np.log(t) / nsamps[ref_arm_idx])
    # Filter arms with UCB value greater than theta
    valid_arms = np.where(I_ucb > lcb_ell)[0]

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

def asymmetric_pe(ref_ell_idx, mu_hat, nsamps, horizon, delta_tilde_ell, delta_tilde, episode_num, last_sampled):
    """
    A function that implements the Asymmetric Pairwise Elimination algorithm from the notes
    https://www.overleaf.com/project/6502fd4306f4b073aa6bd809
    ***********************
    Description identical to that of pairwise_elimination except that we remain a different
     delta_tilde for arm ell and the candidate arm j (and separate underlying implicit round numbers m_\ell and m_j)
    ***********************
    :param ref_ell_idx: Index of the reference arm ell
    :param mu_hat: Empirical estimates of rewards for each candidate arm and arm ell
     Therefore the length of mu_hat is ell
    :param nsamps: Number of times each arm has been sampled
    :param horizon: Known horizon as input
    :param delta_tilde_ell: Algorithm's proxy gap tilde{Delta} for arm ell
    :param delta_tilde: Gaps used by the elimination to set number of samples in a batch and
     UCB buffer terms, reset for both arm ell and candidate arm j at the start of
     every new episode
    :param episode_num: Index of the current episode, varies between 0 and ell - 1
    :param last_sampled: The arm that was being sampled in the previous call to the function since
     the same arm will be sampled until we accumulate n_m samples for it
    :return: Index of the arm to be sampled (k), updated delta_tilde_ell, updated delta_tilde, and updated episode_num
    """
    # If episode ref_ell_idx has been hit in the simulation keep returning ref_ell_idx
    if episode_num == ref_ell_idx:
        return ref_ell_idx, delta_tilde_ell, delta_tilde, episode_num
    # If episode ell has not been hit, then check if the least cost acceptable arm has been identified
    #  as indicated via an invalid episode number of -1
    elif episode_num == -1:
        # In this case the last sampled arm is the arm j that is the successful least cost candidate
        #  and we keep returning it until the horizon budget is exhausted
        return last_sampled, delta_tilde_ell, delta_tilde, episode_num
    # Else if the episode number is a regular valid episode number then assume that the
    #  episode in question is on going
    else:
        # Recompute n_m_ell and n_m for the current episode
        n_m_ell = ceil(2 * log(horizon * delta_tilde_ell ** 2) / (delta_tilde_ell ** 2))
        n_m = ceil(2 * log(horizon * delta_tilde ** 2) / (delta_tilde ** 2))
        # Check the number of times the episode_num `indexed` candidate arm has been sampled
        if nsamps[episode_num] < n_m:
            # If it has not been sampled n_m times, return it again with
            #  all other parameters unchanged
            k = episode_num
            return k, delta_tilde_ell, delta_tilde, episode_num
        # Else arm j (episode_num) must have been sampled n_m times already, so check samples of ell
        elif nsamps[ref_ell_idx] < n_m_ell:
            k = ref_ell_idx
            # If ell has not been sampled n_m times, return it with all other parameters unchanged
            return k, delta_tilde_ell, delta_tilde, episode_num
        # Else both arms have been sampled at least n_m times, so move to the arm elimination phase.
        # conclude the episode and move to the next round within the same episode (if no elimination occurs)
        # or move to the next episode (if an elimination occurs)
        else:
            # Compute the buffer terms for UCB/LCB separately for arm ell and arm j
            buffer_ell = sqrt(log(horizon * delta_tilde_ell ** 2) / (2 * n_m_ell))
            buffer_j = sqrt(log(horizon * delta_tilde ** 2) / (2 * n_m))
            # Check if arm ell should be eliminated in favor of arm j and the episodes concluded
            if mu_hat[ref_ell_idx] + buffer_ell < mu_hat[episode_num] - buffer_j:
                # Set episode to -1 and return arm j for sampling
                k = episode_num
                episode_num = -1
                return k, delta_tilde_ell, delta_tilde, episode_num
            elif mu_hat[episode_num] + buffer_j < mu_hat[ref_ell_idx] - buffer_ell:
                # Go to next episode
                k = episode_num + 1
                # Reset regular delta_tilde for the next episode, but leave delta_tilde_ell unchanged
                delta_tilde = 1.0
                # Increment episode number
                episode_num += 1
                return k, delta_tilde_ell, delta_tilde, episode_num
            else:
                # Continue with the next round of the same episode by updating delta_tilde
                delta_tilde = delta_tilde / 2
                # Check if this update leaves delta_tilde_ell larger than delta_tilde
                if delta_tilde_ell > delta_tilde:
                    # If so, update delta_tilde_ell as well
                    delta_tilde_ell = delta_tilde_ell / 2
                    assert delta_tilde_ell == delta_tilde # Sanity check on the evolution of the two delta_tildes
                # Set the next arm to be sampled to be arm j since we will certainly
                #  need more samples from it whereas, we might not need more samples from ell immediately
                k = episode_num
                return k, delta_tilde_ell, delta_tilde, episode_num


def pe_new(ref_ell_idx, mu_hat, nsamps, horizon, delta_tilde, episode_num, last_sampled):
    """
    A function that implements the Pairwise Elimination algorithm from the notes
    https://www.overleaf.com/project/6502fd4306f4b073aa6bd809
    ***********************
    Through every execution of this function we sample a single arm
    The round number m is not tracked explicitly, but instead it is tracked
     implicitly using the algorithm's proxy gap tilde{Delta}
    Case 1: All the episodes 1 through ell - 1 have been completed
    if so, we simply sample the arm ell until the horizon budget is exhausted
    Case 2: A candidate arm j has already been identified as the least cost acceptable arm
    if so, we keep returning j until the horizon budget is exhausted
    Case 3: An episode j for evaluating the candidacy of arm with index j is ongoing
    if so, we check if the candidate arm j has been sampled n_m times already
    if not, sample arm j until its n_m samples are accumulated
    if yes, check if arm ell has been sampled n_m times already, and sample it if not
    if both arms have been sampled at least n_m times, we move to the arm elimination phase
    Comparisons in lines 14 and 18 of the algo block happen ...
    If the current call of the function was at the cusp of a new episode (candidate arm elimination occurs),
     then in the call we will perform the comparison and if,
     (i) Arm j gets eliminated by arm ell, then we set next arm to be sampled to be j + 1 (j + 1 could equal ell
     which is fine since if we have reached the end of the pack, we sample ell for the entire budget anyway)
     (ii) Arm ell gets eliminated by arm j, then we set next arm to be sampled to be j, set the next episode
     to be -1 to indicate that the least cost acceptable arm has been identified.
    ***********************
    :param ref_ell_idx: Index of the reference arm ell
    :param mu_hat: Empirical estimates of rewards for each candidate arm and arm ell
     Therefore the length of mu_hat is ell
    :param nsamps: Number of times each arm has been sampled
    :param horizon: Known horizon as input
    :param delta_tilde: Gaps used by the elimination to set number of samples in a batch and
     UCB buffer terms, reset for both arm ell and candidate arm j at the start of
     every new episode
    :param episode_num: Index of the current episode, varies between 0 and ell - 1
    :param last_sampled: The arm that was being sampled in the previous call to the function since
     the same arm will be sampled until we accumulate n_m samples for it
    :return: Index of the arm to be sampled (k), updated delta_tilde, and updated episode_num
    """
    # Precompute horizon^e
    horizon_e = horizon ** math.e
    # If episode ref_ell_idx has been hit in the simulation keep returning ref_ell_idx
    if episode_num == ref_ell_idx:
        return ref_ell_idx, delta_tilde, episode_num
    # If episode ell has not been hit, then check if the least cost acceptable arm has been identified
    #  as indicated via an invalid episode number of -1
    elif episode_num == -1:
        # In this case the last sampled arm is the arm j that is the successful least cost candidate
        #  and we keep returning it until the horizon budget is exhausted
        return last_sampled, delta_tilde, episode_num
    # Else if the episode number is a regular valid episode number then assume that the
    #  episode in question is on going
    else:
        # Recompute n_m for the current episode
        n_m = ceil(log(horizon_e * delta_tilde ** 2) / (delta_tilde ** 2))
        # Check the number of times the episode_num indexed candidate arm has been sampled
        if nsamps[episode_num] < n_m:
            # If it has not been sampled n_m times, return it again with
            #  all other parameters unchanged
            k = episode_num
            return k, delta_tilde, episode_num
        # Else arm j (episode_num) must have been sampled n_m times already, so check samples of ell
        elif nsamps[ref_ell_idx] < n_m:
            k = ref_ell_idx
            # If ell has not been sampled n_m times, return it with all other parameters unchanged
            return k, delta_tilde, episode_num
        # Else both arms have been sampled at least n_m times, so move to the arm elimination phase.
        # conclude the episode and move to the next round within the same episode (if no elimination occurs)
        # or move to the next episode (if an elimination occurs)
        else:
            # Compute the buffer terms for UCB/LCB
            buffer = sqrt(log(horizon_e * delta_tilde ** 2) / (2 * n_m))
            # Check if arm ell should be eliminated in favor of arm j and the episodes concluded
            if mu_hat[ref_ell_idx] + buffer < mu_hat[episode_num] - buffer:
                # Set episode to -1 and return arm j for sampling
                k = episode_num
                episode_num = -1
                return k, delta_tilde, episode_num
            elif mu_hat[episode_num] + buffer < mu_hat[ref_ell_idx] - buffer:
                # Go to next episode
                k = episode_num + 1
                # Reset delta_tilde for the next episode
                delta_tilde = 1.0
                # Increment episode number
                episode_num += 1
                return k, delta_tilde, episode_num
            else:
                # Continue with the next round of the same episode by updating delta_tilde
                delta_tilde = delta_tilde / 2
                # Set the next arm to be sampled to be arm j since we will certainly
                #  need more samples from it whereas, we might not need more samples from ell immediately
                k = episode_num
                return k, delta_tilde, episode_num


def pairwise_elimination(ref_ell_idx, mu_hat, nsamps, horizon, delta_tilde, episode_num, last_sampled):
    """
    A function that implements the Pairwise Elimination algorithm from the notes
    https://www.overleaf.com/project/6502fd4306f4b073aa6bd809
    ***********************
    Through every execution of this function we sample a single arm
    The round number m is not tracked explicitly, but instead it is tracked
     implicitly using the algorithm's proxy gap tilde{Delta}
    Case 1: All the episodes 1 through ell - 1 have been completed
    if so, we simply sample the arm ell until the horizon budget is exhausted
    Case 2: A candidate arm j has already been identified as the least cost acceptable arm
    if so, we keep returning j until the horizon budget is exhausted
    Case 3: An episode j for evaluating the candidacy of arm with index j is ongoing
    if so, we check if the candidate arm j has been sampled n_m times already
    if not, sample arm j until its n_m samples are accumulated
    if yes, check if arm ell has been sampled n_m times already, and sample it if not
    if both arms have been sampled at least n_m times, we move to the arm elimination phase
    Comparisons in lines 14 and 18 of the algo block happen ...
    If the current call of the function was at the cusp of a new episode (candidate arm elimination occurs),
     then in the call we will perform the comparison and if,
     (i) Arm j gets eliminated by arm ell, then we set next arm to be sampled to be j + 1 (j + 1 could equal ell
     which is fine since if we have reached the end of the pack, we sample ell for the entire budget anyway)
     (ii) Arm ell gets eliminated by arm j, then we set next arm to be sampled to be j, set the next episode
     to be -1 to indicate that the least cost acceptable arm has been identified.
    ***********************
    :param ref_ell_idx: Index of the reference arm ell
    :param mu_hat: Empirical estimates of rewards for each candidate arm and arm ell
     Therefore the length of mu_hat is ell
    :param nsamps: Number of times each arm has been sampled
    :param horizon: Known horizon as input
    :param delta_tilde: Gaps used by the elimination to set number of samples in a batch and
     UCB buffer terms, reset for both arm ell and candidate arm j at the start of
     every new episode
    :param episode_num: Index of the current episode, varies between 0 and ell - 1
    :param last_sampled: The arm that was being sampled in the previous call to the function since
     the same arm will be sampled until we accumulate n_m samples for it
    :return: Index of the arm to be sampled (k), updated delta_tilde, and updated episode_num
    """
    # If episode ref_ell_idx has been hit in the simulation keep returning ref_ell_idx
    if episode_num == ref_ell_idx:
        return ref_ell_idx, delta_tilde, episode_num
    # If episode ell has not been hit, then check if the least cost acceptable arm has been identified
    #  as indicated via an invalid episode number of -1
    elif episode_num == -1:
        # In this case the last sampled arm is the arm j that is the successful least cost candidate
        #  and we keep returning it until the horizon budget is exhausted
        return last_sampled, delta_tilde, episode_num
    # Else if the episode number is a regular valid episode number then assume that the
    #  episode in question is on going
    else:
        # Recompute n_m for the current episode
        n_m = ceil(2 * log(horizon * delta_tilde ** 2) / (delta_tilde ** 2))
        # Check the number of times the episode_num indexed candidate arm has been sampled
        if nsamps[episode_num] < n_m:
            # If it has not been sampled n_m times, return it again with
            #  all other parameters unchanged
            k = episode_num
            return k, delta_tilde, episode_num
        # Else arm j (episode_num) must have been sampled n_m times already, so check samples of ell
        elif nsamps[ref_ell_idx] < n_m:
            k = ref_ell_idx
            # If ell has not been sampled n_m times, return it with all other parameters unchanged
            return k, delta_tilde, episode_num
        # Else both arms have been sampled at least n_m times, so move to the arm elimination phase.
        # conclude the episode and move to the next round within the same episode (if no elimination occurs)
        # or move to the next episode (if an elimination occurs)
        else:
            # Compute the buffer terms for UCB/LCB
            buffer = sqrt(log(horizon * delta_tilde ** 2) / (2 * n_m))
            # Check if arm ell should be eliminated in favor of arm j and the episodes concluded
            if mu_hat[ref_ell_idx] + buffer < mu_hat[episode_num] - buffer:
                # Set episode to -1 and return arm j for sampling
                k = episode_num
                episode_num = -1
                return k, delta_tilde, episode_num
            elif mu_hat[episode_num] + buffer < mu_hat[ref_ell_idx] - buffer:
                # Go to next episode
                k = episode_num + 1
                # Reset delta_tilde for the next episode
                delta_tilde = 1.0
                # Increment episode number
                episode_num += 1
                return k, delta_tilde, episode_num
            else:
                # Continue with the next round of the same episode by updating delta_tilde
                delta_tilde = delta_tilde / 2
                # Set the next arm to be sampled to be arm j since we will certainly
                #  need more samples from it whereas, we might not need more samples from ell immediately
                k = episode_num
                return k, delta_tilde, episode_num


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
        # Eliminate arms whose UCB has fallen below the largest LCB in the set of active arms by
        #  keeping only the arms whose UCB is greater than the largest LCB
        if ucb_k >= max(active_lcb_list):
            # If not, keep arm k in the new list
            B_new.append(arm_indices_k)

    # Replace the old list with the new list
    B = B_new

    # Update delta_tilde
    delta_tilde = delta_tilde / 2
    # Return package says sample the lowest index arm in the set of active arms
    k = B[0]
    # Return all other parameters
    return k, delta_tilde, B
