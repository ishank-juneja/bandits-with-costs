import numpy as np


def random_argmin(arr):
    """
    A wrapper around np.argmin that breaks ties between minimum index options uniformly at random.

    This function finds the minimum value in the array, identifies all indices where this minimum value occurs,
    and then selects one of these indices at random to break ties uniformly.

    Parameters:
    - arr: A NumPy array of any shape.

    Returns:
    - int: The index of the first occurrence of the minimum value in the array, with ties broken randomly.
    """
    # Directly use boolean indexing to find indices of the minimum value
    min_indices = np.flatnonzero(arr == np.min(arr))

    # Select one of these indices at random and return
    return np.random.choice(min_indices)


def random_argmax(arr):
    """
    Equivalent to random_argmin, but for argmax instead
    :param arr:
    :return:
    """
    max_indices = np.flatnonzero(arr == np.max(arr))

    return np.random.choice(max_indices)
