import numpy as np
from math import log, sqrt


# pestimates are empirical estimate of probabilities
# nsamps is number of times each arm is sampled
def UCB(p_estimates, nsamps, t):
    # Update ucb value
    I_ucb = p_estimates + np.sqrt(2 * log(t) / nsamps)
    # Determine arm to be sampled in current step,
    # Ties are broken by lower index preference
    k = np.argmax(I_ucb)
    return k


def UCB_CS(p_estimates, nsamps, t, costs, theta):
    """
    p_estimates: Array of estimated success probabilities for each arm
    nsamps: Array of number of times each arm has been pulled up to time t-1
    t: Current time step
    costs: Array of cost associated with pulling each arm
    theta: Threshold value
    """

    # Compute the upper confidence bounds for all arms
    I_ucb = p_estimates + np.sqrt(8 * np.log(t) / nsamps)

    # Filter arms with UCB value greater than theta
    valid_arms = np.where(I_ucb > theta)[0]

    # If no arm has UCB greater than theta, return a random arm
    if len(valid_arms) == 0:
        return np.random.choice(range(len(p_estimates)))

    # Among the valid arms, select the arm with the smallest cost
    k = valid_arms[np.argmin(costs[valid_arms])]

    return k


# Library only
if __name__ == '__main__':
    exit(0)
