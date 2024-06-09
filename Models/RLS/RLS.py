"""
Recursive Least Squares (RLS) Script

This script contains functions for implementing the Recursive Least Squares (RLS) algorithm and related analysis
for linear regression tasks. It utilizes scikit-learn for K-Fold cross-validation and various utility functions for
measures,
predictions, plotting, and constant definitions.

Functions:
- `rls`: Implement the Recursive Least Squares algorithm for optimizing linear regression coefficients.
- `rls_KFold`: Perform K-Fold cross-validation with Recursive Least Squares for linear regression.
- `rls_adversarial`: Evaluate the Recursive Least Squares algorithm on adversarial test data.
- `rls_convergence`: Perform Recursive Least Squares for linear regression with convergence analysis.
- `rls_plot_convergence`: Perform Recursive Least Squares with convergence analysis and plot results.

Dependencies:
- `numpy`: Numerical library for array manipulation and calculations.
- `sklearn.model_selection.KFold`: K-Fold cross-validation for splitting data into training and test sets.
- `Utils`: Custom utility module providing measures, predictions, plotting, and constant definitions.
- `Measures`: Utility functions for evaluating performance measures like R-squared.
- `Predictions`: Utility functions for computing predictions using model coefficients.
- `Util`: Utility functions for data manipulation and analysis.
- `Plotter`: Utility functions for creating plots and visualizing data.
- `Constants`: Module containing constant definitions for paths and configurations.


Author: M. Shaira
Date: Aug, 2023
"""
import numpy as np
from sklearn.model_selection import KFold
from Utils import Measures, Predictions, Util, Plotter, Constants


def rls(X, y, lambda_, delta, modular):
    """
    Implement the Recursive Least Squares algorithm for optimizing linear regression coefficients.

    Args:
        X (array-like): Input feature matrix.
        y (array-like): Target values.
        lambda_ (float): Forgetting factor, controls the influence of previous data.
        delta (float): Initial value for the covariance matrix P.

    Returns:
        w (array): Optimized coefficient vector.
        epoch_list (array): List of epoch indices.
        cost_list (array): List of corresponding costs.

    """
    num_samples, num_features = X.shape
    w = np.zeros(num_features)
    P = delta * np.eye(num_features)

    cost_list = np.array([])
    epoch_list = np.array([])

    for t in range(len(X)):
        x_t = X[t, :].reshape(-1, 1)
        y_t = y[t]

        error = y_t - np.dot(x_t.T, w)
        K = np.dot(P, x_t) / (lambda_ + np.dot(np.dot(x_t.T, P), x_t))
        w += np.dot(K, error)
        P = (P - np.dot(np.dot(K, x_t.T), P)) / lambda_

        y_predicted = np.dot(w, x_t)
        cost = np.square(y_t - y_predicted)

        if t % modular == 0:
            cost_list = np.append(cost_list, cost)
            epoch_list = np.append(epoch_list, t)

    return w, epoch_list, cost_list


def rls_KFold(X, y, lambda_, delta, seed, modular):
    """
        Perform K-Fold cross-validation with Recursive Least Squares for linear regression.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            lambda_ (float): Forgetting factor, controls the influence of previous data.
            delta (float): Initial value for the covariance matrix P.
            seed (int): Random seed for reproducibility.

        Returns:
            acc (float): Mean accuracy (R-squared) across K-Fold splits.

        """
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    costs = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, epoch_list, cost_list = rls(X_train, y_train, lambda_, delta, modular)
        predicted_y_test = Predictions.compute_predictions(X_test, w)
        acc_per_split_for_same_seed = Measures.r2_score_(y_test, predicted_y_test)
        scores.append(acc_per_split_for_same_seed)
        avg_cost = np.array(cost_list).mean()
        costs.append(avg_cost)


    return np.array(scores).mean(), np.array(costs).mean()


def rls_adversarial(X_train, y_train, X_test, y_test, lambda_, delta):
    """
        Evaluate the Recursive Least Squares algorithm on adversarial test data.

        Args:
            X_train (array-like): Training input feature matrix.
            y_train (array-like): Training target values.
            X_test (array-like): Adversarial test input feature matrix.
            y_test (array-like): Adversarial test target values.
            lambda_ (float): Forgetting factor, controls the influence of previous data.
            delta (float): Initial value for the covariance matrix P.

        Returns:
            acc (float): Accuracy (R-squared) on adversarial test data.

        """
    w, epoch_list, cost_list = rls(X_train, y_train, lambda_, delta)
    predicted_y_test = Predictions.compute_predictions(X_test, w)
    acc = Measures.r2_score_(y_test, predicted_y_test)
    return acc


def rls_convergence(X, y, lambda_, delta):
    """
        Perform Recursive Least Squares for linear regression with convergence analysis.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            lambda_ (float): Forgetting factor (hyperparameter), controls the influence of previous data.
            delta (float): Initial value for the covariance matrix P.

        Returns:
            acc (float): Mean accuracy (R-squared) across K-Fold splits.
            epochs_accu (array): Accumulated epochs divided by number of splits.
            cost_accu (array): Accumulated costs divided by number of splits.

        """
    n_splits = 5
    kf = KFold(n_splits)
    scores = []
    epoch_list_per_seed = np.array([])
    cost_list_per_seed = np.array([])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, epoch_list, cost_list = rls(X_train, y_train, lambda_, delta)
        epoch_list_per_seed = Util.sum_lists_element_wise(epoch_list_per_seed, epoch_list)
        cost_list_per_seed = Util.sum_lists_element_wise(cost_list_per_seed, cost_list)
        predicted_y_test = Predictions.compute_predictions(X_test, w)
        acc_per_split_for_same_seed = Measures.r2_score_(y_test, predicted_y_test)
        scores.append(acc_per_split_for_same_seed)

    acc = np.array(scores).mean()
    epochs_accu = epoch_list_per_seed / n_splits
    cost_accu = cost_list_per_seed / n_splits
    return acc, epochs_accu, cost_accu


def rls_plot_convergence(X, y, lambda_, delta, X_test, y_test, model_name):
    """
        Perform Recursive Least Squares with convergence analysis and plot results.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            lambda_ (float): Forgetting factor (hyperparameter), controls the influence of previous data.
            delta (float): Initial value for the covariance matrix P.
            X_test (array-like): Test input feature matrix.
            y_test (array-like): Test target values.
            model_name (str): Name of the model for plotting purposes.

        Returns:
            w (array): Optimized coefficient vector.

        """
    Util.create_directory(Constants.plotting_path + model_name)
    num_samples, num_features = X.shape
    w = np.zeros(num_features)
    P = delta * np.eye(num_features)

    accumulated_xs = []
    accumulated_ys = []
    for i in range(X.shape[0]):
        x_t = X[i, :].reshape(-1, 1)
        y_t = y[i]
        accumulated_xs.append(x_t)
        accumulated_ys.append(y_t)
        error = y_t - np.dot(x_t.T, w)
        K = np.dot(P, x_t) / (lambda_ + np.dot(np.dot(x_t.T, P), x_t))
        w += np.dot(K, error)
        P = (P - np.dot(np.dot(K, x_t.T), P)) / lambda_

        Plotter.compute_acc_plot_per_iteration(X_train=X, y_train=y, w=w, b=None,
                                               iteration=i, X_test=X_test, y_test=y_test,
                                               accumulated_xs=accumulated_xs, accumulated_ys=accumulated_ys,
                                               model_name=model_name)
    return w



def rls_plot_convergence2(X, y, lambda_, delta, X_test, y_test, model_name, modular):
    """
        Perform Recursive Least Squares with convergence analysis and plot results.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            lambda_ (float): Forgetting factor (hyperparameter), controls the influence of previous data.
            delta (float): Initial value for the covariance matrix P.
            X_test (array-like): Test input feature matrix.
            y_test (array-like): Test target values.
            model_name (str): Name of the model for plotting purposes.

        Returns:
            w (array): Optimized coefficient vector.

        """
    num_samples, num_features = X.shape
    w = np.zeros(num_features)
    P = delta * np.eye(num_features)
    accumulated_size = 0
    rls_map = {}
    rls_mse_map = {}
    for i in range(X.shape[0]):
        x_t = X[i, :].reshape(-1, 1)
        y_t = y[i]
        accumulated_size +=1
        error = y_t - np.dot(x_t.T, w)
        K = np.dot(P, x_t) / (lambda_ + np.dot(np.dot(x_t.T, P), x_t))
        w += np.dot(K, error)
        P = (P - np.dot(np.dot(K, x_t.T), P)) / lambda_

        if accumulated_size % modular == 0:
            y_predicted = Predictions.compute_predictions(X_test, w)
            acc = Measures.r2_score_(y_test, y_predicted)
            rls_map[accumulated_size] = acc

            mse = np.mean((y_test - y_predicted)**2)
            rls_mse_map[accumulated_size] = mse

    return w, rls_map, rls_mse_map


def rls_convergence2(X, y, lambda_, delta, model_name, seed, modular):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    rls_list = []
    rls_mse_list = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, rls_map, rls_mse_map = rls_plot_convergence2(X, y, lambda_, delta, X_test, y_test, model_name, modular)
        rls_list.append(rls_map)
        rls_mse_list.append(rls_mse_map)

    return rls_list, rls_mse_list