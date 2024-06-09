import numpy as np
from scipy.optimize import minimize


def linear_regression_mle(x_train, y_train):
    # Define negative log-likelihood function
    def negative_log_likelihood(theta):
        # Extract parameters
        beta = theta[:-1]  # coefficients
        bias = theta[-1]   # bias term
        sigma = 1.0        # fixed for simplicity, can be estimated as well

        # Predictions
        y_pred = np.dot(x_train, beta) + bias

        # Negative log-likelihood
        error = y_train - y_pred
        L = np.sum(0.5 * np.log(2 * np.pi * sigma ** 2) + (error ** 2) / (2 * sigma ** 2))
        return L

    # Initial guess for parameters
    n_features = x_train.shape[1]
    initial_guess = np.random.randn(n_features + 1)

    # Minimize negative log-likelihood using L-BFGS-B optimization
    MLE = minimize(negative_log_likelihood, initial_guess, method='L-BFGS-B')

    # Extract parameters
    parameters = MLE['x']
    beta = parameters[:-1]
    bias = parameters[-1]

    return beta, bias