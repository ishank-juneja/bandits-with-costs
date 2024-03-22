# Add an algorithm here that solves the full cost subsidy problem
#  using an Improved UCB as the BAI approach in phase 1 and
#  the pairwise elimination type scheme in the second phase to solve the problem


def CS_ETC(p_estimates, nsamps, t, delta, mu_o, alpha):
    """
    Implementation of the CS-ETC algorithm from the MAB-CS paper
    :param p_estimates: array to hold mu_hat
    :param nsamps: Array to hold the number of times each arm has been sampled
    :param t:
    :param mu_o:
    :param alpha:
    :return:
    """
