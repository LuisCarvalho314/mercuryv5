import numpy as np


def compute_cooccurrence_matrix(z_hat, z_true):
    """
    Compute the co-occurrence matrix C[i, j] = number of times z_hat[n] = i and z_true[n] = j,
    where z_hat and z_true contain string labels.
    """
    assert len(z_hat) == len(z_true), "Sequences must have the same length"

    # Get sorted unique labels
    hat_labels = sorted(set(z_hat))
    true_labels = sorted(set(z_true))

    # Create label-to-index mappings
    hat_to_index = {label: i for i, label in enumerate(hat_labels)}
    true_to_index = {label: j for j, label in enumerate(true_labels)}

    # Initialize a proper 2D NumPy array
    C = np.zeros((len(hat_labels), len(true_labels)), dtype=int)

    for zh, zt in zip(z_hat, z_true):
        i = hat_to_index[zh]
        j = true_to_index[zt]
        C[i, j] += 1

    return C  # Just return the NumPy array if the rest of your code expects that


def compute_precision(z_hat, z_true):
    """
    Compute the precision metric based on the co-occurrence matrix.

    Returns:
        precision (float): Value between 0 and 1
    """
    C = compute_cooccurrence_matrix(z_hat, z_true)
    correct_assignments = np.sum(np.max(C, axis=1))  # max over each row
    total = len(z_hat)
    return correct_assignments / total
