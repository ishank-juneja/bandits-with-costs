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


def pairwise_elimination_for_cs_pe(ref_ell_idx: int, mu_hat: np.array, nsamps: np.array, horizon: int,
                                   delta_tilde: float, episode_num: int, last_sampled: int, omega: np.array, alpha=0.0):
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
    :param mu_hat: Array of floats Empirical estimates of rewards for each candidate arm and arm ell
     Therefore the length of mu_hat is ell
    :param nsamps: Array of ints Number of times each arm has been sampled
    :param horizon: Known horizon as input
    :param delta_tilde: Gaps used by the elimination to set number of samples in a batch and
     UCB buffer terms, reset for both arm ell and candidate arm j at the start of
     every new episode
    :param episode_num: Index of the current episode, varies between 0 and ell - 1
    :param last_sampled: The arm that was being sampled in the previous call to the function since
     the same arm will be sampled until we accumulate n_m samples for it
    :return: Index of the arm to be sampled (k), updated delta_tilde, and updated episode_num
    """
    # Reference arm reward multiplier
    ref_rew_multiplier = 1 - alpha
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
        # Recompute n_m for the current round of the current episode
        n_m = ceil(2 * log(horizon * delta_tilde ** 2) / (delta_tilde ** 2))
        # Recompute the buffer terms for UCB/LCB on every function call
        buffer = sqrt(log(horizon * delta_tilde ** 2) / (2 * n_m))
        # At the start, if we have the number of samples of the candidate arm as
        #  equal to n_m, then it means that no further samples are needed before the next
        #  elimination check and round number increment
        # Check if arm ell should be eliminated in favor of arm j and the episodes concluded
        if (nsamps[episode_num] == n_m) and (ref_rew_multiplier * mu_hat[ref_ell_idx] + buffer < mu_hat[episode_num] - buffer):
            # Set episode to -1 and return arm j for sampling
            k = episode_num
            episode_num = -1
            return k, delta_tilde, episode_num
        elif (nsamps[episode_num] == n_m) and (mu_hat[episode_num] + buffer < ref_rew_multiplier * mu_hat[ref_ell_idx] - buffer):
            # Go to next episode
            k = episode_num + 1
            # Increment episode number
            episode_num += 1
            # Reset delta_tilde for the next episode
            delta_tilde = omega[episode_num]
            return k, delta_tilde, episode_num
        # Check the number of times the episode_num indexed candidate arm has been sampled
        elif nsamps[episode_num] < n_m:
            # If it has not been sampled n_m times, return it again with
            #  all other parameters unchanged
            k = episode_num
            return k, delta_tilde, episode_num
        # Else arm j (episode_num) must have been sampled n_m times already, so check samples of ell
        elif nsamps[ref_ell_idx] < n_m:
            k = ref_ell_idx
            # If ell has not been sampled n_m times, return it with all other parameters unchanged
            return k, delta_tilde, episode_num
        else:
            # Continue with the next round of the same episode by updating delta_tilde
            delta_tilde = delta_tilde / 2
            # Set the next arm to be sampled to be arm j since we will certainly
            #  need more samples from it whereas, we might not need more samples from ell immediately
            k = episode_num
            return k, delta_tilde, episode_num


def pairwise_elimination(ref_ell_idx: int, mu_hat: np.array, nsamps: np.array, horizon: int, delta_tilde: float,
                         episode_num: int, last_sampled: int, alpha: float=0.0):
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
    # Reference arm reward multiplier
    ref_rew_multiplier = 1 - alpha
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
            if ref_rew_multiplier * mu_hat[ref_ell_idx] + buffer < mu_hat[episode_num] - buffer:
                # Set episode to -1 and return arm j for sampling
                k = episode_num
                episode_num = -1
                return k, delta_tilde, episode_num
            elif mu_hat[episode_num] + buffer < ref_rew_multiplier * mu_hat[ref_ell_idx] - buffer:
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
