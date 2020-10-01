import numpy as np

def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    for i in range(len(w0)):
        for j in range(len(w1)):
            w = np.array((w0[i], w1[j]))
            losses[i,j] = compute_loss(y, tx, w)
    return losses

def standardize(x):
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    return std_data

def standardize_with_meanstd(x, mean, std):
    centered_data = x - mean
    std_data = centered_data / std
    return std_data