import numpy as np
from math import log


# pestimates are empirical estimate of probabilities
# nsamps is number of times each arm is sampled
def UCB(p_estimates, nsamps, t):
    # Update ucb value
    I_ucb = p_estimates + np.sqrt(2 * log(t) / nsamps)
    # Determine arm to be sampled in current step,
    # Ties are broken by lower index preference
    k = np.argmax(I_ucb)
    return k
