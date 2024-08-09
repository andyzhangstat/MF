import numpy as np






def matrix_factorization_als(Y_train, Y_test, r=15, lambda_reg=5, iterations=30):
    m, n = Y_train.shape

    # Initial value for X and Y
    np.random.seed(123)
    X = np.random.randn(m, r) + 0.1
    Y = np.random.randn(n, r) + 0.1

    lambda_eye = lambda_reg * np.eye(r)  # Precompute regularization term
    RMSE_train = np.zeros(iterations)
    RMSE_test = np.zeros(iterations)

    for itr in range(iterations):
        R = X @ Y.T

        # Calculate RMSE for training and test data
        RMSE_train[itr] = np.sqrt(np.nansum((Y_train - R) ** 2) / np.count_nonzero(~np.isnan(Y_train)))
        RMSE_test[itr] = np.sqrt(np.nansum((Y_test - R) ** 2) / np.count_nonzero(~np.isnan(Y_test)))

        for i in range(m):
            Ri = np.nonzero(~np.isnan(Y_train[i, :]))[0]
            if Ri.size > 0:
                Xi = np.linalg.solve(Y[Ri, :].T @ Y[Ri, :] + lambda_eye, Y_train[i, Ri] @ Y[Ri, :])
                X[i, :] = Xi

        for j in range(n):
            Rj = np.nonzero(~np.isnan(Y_train[:, j]))[0]
            if Rj.size > 0:
                Yj = np.linalg.solve(X[Rj, :].T @ X[Rj, :] + lambda_eye, Y_train[Rj, j] @ X[Rj, :])
                Y[j, :] = Yj


    return X, Y, RMSE_train, RMSE_test






def matrix_factorization_gd(Y_train, Y_test, r=20, eta=0.005, lambda_reg=0.1, iterations=50):
    m, n = Y_train.shape

    # Initialization
    X = np.random.randn(m, r) * 0.01
    Y = np.random.randn(n, r) * 0.01
    RMSE_train = np.zeros(iterations)
    RMSE_test = np.zeros(iterations)

    for itr in range(iterations):
        R = X @ Y.T

        # Calculate RMSE for training and test data
        RMSE_train[itr] = np.sqrt(np.nansum((Y_train - R) ** 2) / np.count_nonzero(~np.isnan(Y_train)))
        RMSE_test[itr] = np.sqrt(np.nansum((Y_test - R) ** 2) / np.count_nonzero(~np.isnan(Y_test)))

        # Update X
        for i in range(m):
            Ri = np.nonzero(~np.isnan(Y_train[i, :]))[0]
            if Ri.size > 0:
                diff = X[i, :] @ Y[Ri, :].T - Y_train[i, Ri]
                gradient = lambda_reg * X[i, :] + diff @ Y[Ri, :]
                X[i, :] -= eta * gradient

        # Update Y
        for j in range(n):
            Rj = np.nonzero(~np.isnan(Y_train[:, j]))[0]
            if Rj.size > 0:
                diff = Y[j, :] @ X[Rj, :].T - Y_train[Rj, j]
                gradient = lambda_reg * Y[j, :] + diff @ X[Rj, :]
                Y[j, :] -= eta * gradient

    return X, Y, RMSE_train, RMSE_test






def matrix_factorization_sgd(Y_train, Y_test, train_data, r=15, eta=0.01, lambda_reg=0.1, iterations=50):
    m, n = Y_train.shape

    # Initialize matrices X and Y
    X = np.random.randn(m, r) * 0.01
    Y = np.random.randn(n, r) * 0.01
    RMSE_train = np.zeros(iterations)
    RMSE_test = np.zeros(iterations)

    for itr in range(iterations):
        # Compute predictions
        R = X @ Y.T
        
        # Calculate RMSE for training and test data
        RMSE_train[itr] = np.sqrt(np.nansum((Y_train - R) ** 2) / np.count_nonzero(~np.isnan(Y_train)))
        RMSE_test[itr] = np.sqrt(np.nansum((Y_test - R) ** 2) / np.count_nonzero(~np.isnan(Y_test)))

        # Shuffle indices for stochastic gradient descent
        np.random.shuffle(train_data)

        # Perform SGD updates
        for idx in train_data:
            i, j = idx

            if Y_train[i, j] != 0:  # Only update for non-zero entries in Y_train
                tmp = X[i, :] @ Y[j, :].T - Y_train[i, j]
                X[i, :] -= eta * (lambda_reg * X[i, :] + tmp * Y[j, :])
                Y[j, :] -= eta * (lambda_reg * Y[j, :] + tmp * X[i, :])

    return X, Y, RMSE_train, RMSE_test

