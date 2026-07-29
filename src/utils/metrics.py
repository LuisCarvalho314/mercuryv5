import numpy as np


def compute_cooccurrence_matrix(z_hat, z_true):
    """
    Compute the co-occurrence matrix C[i, j] = number of times z_hat[n] = i and z_true[n] = j,
    where z_hat and z_true contain string labels.
    """
    assert len(z_hat) == len(z_true), "Sequences must have the same length"
    if len(z_hat) == 0:
        return np.zeros((0, 0), dtype=int)

    _, hat_inverse = np.unique(np.asarray(z_hat), return_inverse=True)
    _, true_inverse = np.unique(np.asarray(z_true), return_inverse=True)
    C = np.zeros((int(hat_inverse.max()) + 1, int(true_inverse.max()) + 1), dtype=int)
    np.add.at(C, (hat_inverse, true_inverse), 1)

    return C  # Just return the NumPy array if the rest of your code expects that


def compute_precision(z_hat, z_true):
    """
    Compute the precision metric based on the co-occurrence matrix.

    Returns:
        precision (float): Value between 0 and 1
    """
    C = compute_cooccurrence_matrix(z_hat, z_true)
    if C.size == 0:
        return 0.0
    correct_assignments = np.sum(np.max(C, axis=1))  # max over each row
    total = len(z_hat)
    return correct_assignments / total
