import numpy as np

def simulate_matrix_data(seed=2018, m=100, n=100, J=5, noise_variance=20, missfrac=0.8, testfrac=0.2):
    """
    Simulate a matrix with a given rank and noise, split into training and test sets.

    Parameters:
    - seed: Random seed for reproducibility.
    - m: Number of rows in the matrix.
    - n: Number of columns in the matrix.
    - J: Rank of the underlying matrix.
    - noise_variance: Variance of the Gaussian noise added to the matrix.
    - missfrac: Fraction of missing entries in the partially observed matrix.
    - testfrac: Fraction of observed entries to be used for testing.

    Returns:
    - Y_complete: The fully observed matrix.
    - Y_train: The training matrix with missing test entries.
    - Y_test: The test matrix with only the test entries.
    """
    np.random.seed(seed)

    # Generate low-rank matrices U and V
    U = np.random.randn(m, J)
    V = np.random.randn(n, J)
    # Generate the fully observed matrix with noise
    B_star = U @ V.T
    Y_complete = B_star + np.random.normal(0, np.sqrt(noise_variance), (m, n))

    # Create the partially observed matrix
    imiss = np.random.choice(m * n, int(m * n * missfrac), replace=False)
    Y = Y_complete.copy()
    Y.ravel()[imiss] = np.nan

    # Find the indices of observed entries
    observed_indices = np.array(np.where(~np.isnan(Y))).T
    np.random.shuffle(observed_indices)

    # Split observed indices into training and test sets
    num_test = int(len(observed_indices) * testfrac)
    test_indices = observed_indices[:num_test]
    train_indices = observed_indices[num_test:]

    # Create training and test matrices
    Y_train = np.full(Y.shape, np.nan)
    Y_test = np.full(Y.shape, np.nan)

    # Fill in the training and test matrices
    Y_train[tuple(train_indices.T)] = Y[tuple(train_indices.T)]
    Y_test[tuple(test_indices.T)] = Y[tuple(test_indices.T)]

    return Y_complete, Y_train, Y_test






