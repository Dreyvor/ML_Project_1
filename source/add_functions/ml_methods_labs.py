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
    # if bias column: 
    if np.array_equal(x[:, 0], np.ones(len(x))):
        centered_data = x[:,1:] - np.mean(x[:,1:], axis=0)
        std_data = centered_data / np.std(centered_data, axis=0)
        return np.hstack((np.ones((len(x),1)),std_data))
    else:
        centered_data = x - np.mean(x, axis=0)
        std_data = centered_data / np.std(centered_data, axis=0)
        return std_data

def standardize_with_mean_std(x, mean, std):
    # if bias term:
    if np.array_equal(x[:, 0], np.ones(len(x))):
        std_data = (x[:,1:] - mean)/std
        return np.hstack((np.ones((len(x),1)), std_data))
    else:
        std_data = (x - mean)/std
        return std_data