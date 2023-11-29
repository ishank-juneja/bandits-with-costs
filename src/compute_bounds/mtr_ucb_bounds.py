import math


def expected_pulls(n, delta_mu_i):
    """
    Compute the upper bound of the expected number of pulls.

    Parameters:
    n (int): The number of trials or iterations.
    delta_mu_i (float): The gap or difference parameter.

    Returns:
    float: The upper bound of the expected number of pulls.
    """
    if delta_mu_i == 0:
        raise ValueError("delta_mu_i must be non-zero to avoid division by zero")

    term1 = (8 * math.log(n)) / (delta_mu_i ** 2)
    term2 = 1
    term3 = math.pi**2 / 3

    return term1 + term2 + term3


def expected_pulls_j(n):
    """
    Compute the upper bound of the expected number of pulls for T_j(n).

    Parameter:
    n (int): The number of trials or iterations. Not used in calculation but included for consistency.

    Returns:
    float: The upper bound of the expected number of pulls for T_j(n).
    """
    term1 = 1
    term2 = math.pi**2 / 3

    return term1 + term2


# Write a quick test to make sure the function works
if __name__ == "__main__":
    n = 2e6
    delta_mu_i = 0.01
    print(expected_pulls(n, delta_mu_i))
    print(expected_pulls_j(n))
