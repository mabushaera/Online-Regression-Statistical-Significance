"""
Online Ridge Regression Script

This script contains functions for performing Online Ridge Regression, a variant of linear regression,
where samples are processed one by one in an online manner. It utilizes scikit-learn for K-Fold cross-validation
and various utility functions for measures, predictions, plotting, and constant definitions.

Functions:
- `online_ridge_regression`: Implement the Online Ridge Regression algorithm for linear regression.
- `online_ridge_regression_KFold`: Perform K-Fold cross-validation with Online Ridge Regression.
- `online_ridge_regression_adversarial`: Evaluate Online Ridge Regression on adversarial test data.
- `online_ridge_regression_convergence`: Perform Online Ridge Regression with convergence analysis.
- `online_ridge_regression_plot_convergence`: Perform Online Ridge Regression with convergence analysis and plot results.

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


def online_ridge_regression(X, y, learning_rate, epochs, regularization_param, modular):
    """
        Implement the Online Ridge Regression algorithm for linear regression.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            learning_rate (float): Learning rate controlling parameter updates.
            epochs (int): Number of epochs (iterations).
            regularization_param (float): Regularization parameter for controlling bias-variance trade-off.

        Returns:
            w (array): Optimized coefficient vector.
            bias (float): Bias term.
            epoch_list (array): List of epoch indices.
            cost_list (array): List of corresponding costs.

        """
    num_of_samples, num_of_features = X.shape
    w = np.zeros(num_of_features)
    bias = 0

    accumulated_xs = []  # this will append each single xs on each iteration for plotting reasons
    accumulated_ys = []  # this will append each single ys on each iteration for plotting reasons

    cost_list = np.array([])
    epoch_list = np.array([])

    for i in range(epochs):
        index = np.random.randint(num_of_samples)
        x_sample = X[index]
        y_sample = y[index]

        accumulated_xs.append(x_sample)
        accumulated_ys.append(y_sample)

        y_predicted = np.dot(x_sample, w.T) + bias

        dw = 2 * x_sample * (y_predicted - y_sample) + 2 * regularization_param * w
        db = 2 * (y_predicted - y_sample)

        w -= learning_rate * dw
        bias -= learning_rate * db

        cost = np.square(y_sample - y_predicted)

        if i % modular == 0:  # at every 50 iteration record the cost and epoch value
            cost_list = np.append(cost_list, cost)
            epoch_list = np.append(epoch_list, i)

    return w, bias, epoch_list, cost_list


def online_ridge_regression_KFold(X, y, learning_rate, epochs, regularization_param, seed, modular):
    """
        Perform K-Fold cross-validation with Online Ridge Regression.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            learning_rate (float): Learning rate controlling parameter updates.
            epochs (int): Number of epochs (iterations).
            regularization_param (float): Regularization parameter for controlling bias-variance trade-off.
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
        w, b, epoch_list, cost_list = online_ridge_regression(X_train, y_train, learning_rate,
                                                              epochs, regularization_param, modular)
        predicted_y_test = Predictions.compute_predictions_(X_test, w, b)
        acc_per_split_for_same_seed = Measures.r2_score_(y_test, predicted_y_test)
        scores.append(acc_per_split_for_same_seed)
        avg_cost = np.array(cost_list).mean()
        costs.append(avg_cost)


    return np.array(scores).mean(), np.array(costs).mean()


def online_ridge_regression_adversarial(X_train, y_train, X_test, y_test, learning_rate, epochs, regularization_param):
    """
        Evaluate Online Ridge Regression on adversarial test data.

        Args:
            X_train (array-like): Training input feature matrix.
            y_train (array-like): Training target values.
            X_test (array-like): Adversarial test input feature matrix.
            y_test (array-like): Adversarial test target values.
            learning_rate (float): Learning rate controlling parameter updates.
            epochs (int): Number of epochs (iterations).
            regularization_param (float): Regularization parameter for controlling bias-variance trade-off.

        Returns:
            acc (float): Accuracy (R-squared) on adversarial test data.

        """
    w, b, epoch_list, cost_list = online_ridge_regression(X_train, y_train, learning_rate, epochs,
                                                          regularization_param)
    predicted_y_test = Predictions.compute_predictions_(X_test, w, b)
    acc = Measures.r2_score_(y_test, predicted_y_test)

    return acc


def online_ridge_regression_convergence(X, y, learning_rate, epochs, regularization_param):
    """
        Perform Online Ridge Regression with convergence analysis.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            learning_rate (float): Learning rate controlling parameter updates.
            epochs (int): Number of epochs (iterations).
            regularization_param (float): Regularization parameter for controlling bias-variance trade-off.

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
        w, b, epoch_list, cost_list = online_ridge_regression(X_train, y_train, learning_rate,
                                                              epochs, regularization_param)

        epoch_list_per_seed = Util.sum_lists_element_wise(epoch_list_per_seed, epoch_list)
        cost_list_per_seed = Util.sum_lists_element_wise(cost_list_per_seed, cost_list)
        predicted_y_test = Predictions.compute_predictions_(X_test, w, b)
        acc_per_split_for_same_seed = Measures.r2_score_(y_test, predicted_y_test)
        scores.append(acc_per_split_for_same_seed)

    acc = np.array(scores).mean()
    epochs_accu = epoch_list_per_seed / n_splits
    cost_accu = cost_list_per_seed / n_splits

    return acc, epochs_accu, cost_accu


def online_ridge_regression_plot_convergence(X, y, learning_rate, epochs, regularization_param, X_test, y_test,
                                             model_name):
    """
        Perform Online Ridge Regression with convergence analysis and plot results.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            learning_rate (float): Learning rate controlling parameter updates.
            epochs (int): Number of epochs (iterations).
            regularization_param (float): Regularization parameter for controlling bias-variance trade-off.
            X_test (array-like): Test input feature matrix.
            y_test (array-like): Test target values.
            model_name (str): Name of the model for plotting purposes.

        Returns:
            w (array): Optimized coefficient vector.
            bias (float): Bias term.

        """
    Util.create_directory(Constants.plotting_path + model_name)
    num_of_samples, num_of_features = X.shape
    w = np.zeros(num_of_features)
    bias = 0

    accumulated_xs = []  # this will append each single xs on each iteration for plotting reasons
    accumulated_ys = []  # this will append each single ys on each iteration for plotting reasons

    for i in range(epochs):
        index = np.random.randint(num_of_samples)
        x_sample = X[index]
        y_sample = y[index]

        accumulated_xs.append(x_sample)
        accumulated_ys.append(y_sample)

        y_predicted = np.dot(x_sample, w.T) + bias

        dw = 2 * x_sample * (y_predicted - y_sample) + 2 * regularization_param * w
        db = 2 * (y_predicted - y_sample)

        w -= learning_rate * dw
        bias -= learning_rate * db

        Plotter.compute_acc_plot_per_iteration(X_train=X, y_train=y, w=w, b=bias,
                                               iteration=i, X_test=X_test, y_test=y_test,
                                               accumulated_xs=accumulated_xs, accumulated_ys=accumulated_ys,
                                               model_name=model_name)

    return w, bias



def online_ridge_regression_plot_convergence2(X, y, learning_rate, epochs, regularization_param, X_test, y_test,
                                             model_name, modular):
    """
        Perform Online Ridge Regression with convergence analysis and plot results.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            learning_rate (float): Learning rate controlling parameter updates.
            epochs (int): Number of epochs (iterations).
            regularization_param (float): Regularization parameter for controlling bias-variance trade-off.
            X_test (array-like): Test input feature matrix.
            y_test (array-like): Test target values.
            model_name (str): Name of the model for plotting purposes.

        Returns:
            w (array): Optimized coefficient vector.
            bias (float): Bias term.

        """
    num_of_samples, num_of_features = X.shape
    w = np.zeros(num_of_features)
    bias = 0

    accumulated_size = 0
    orr_map = {}
    orr_mse_map = {}
    for i in range(epochs):
        index = np.random.randint(num_of_samples)
        x_sample = X[index]
        y_sample = y[index]

        accumulated_size +=1

        y_predicted = np.dot(x_sample, w.T) + bias

        dw = 2 * x_sample * (y_predicted - y_sample) + 2 * regularization_param * w
        db = 2 * (y_predicted - y_sample)

        w -= learning_rate * dw
        bias -= learning_rate * db

        if(accumulated_size%modular ==0):
            y_predicted = Predictions.compute_predictions_(X_test, w, bias)
            acc = Measures.r2_score_(y_test, y_predicted)
            orr_map[accumulated_size] = acc

            mse = np.mean((y_test - y_predicted)**2)
            orr_mse_map[accumulated_size] = mse

    return w, bias, orr_map, orr_mse_map


def online_ridge_regression_convergence2(X, y,
                                         learning_rate,
                                         epochs,
                                         regularization_param,
                                         model_name, seed, modular):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    orr_list = []
    orr_mse_list = []
    for fold_index, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, b, orr_map, orr_mse_map = online_ridge_regression_plot_convergence2(X_train, y_train, learning_rate, epochs, regularization_param, X_test, y_test, model_name, modular)
        orr_list.append(orr_map)
        orr_mse_list.append(orr_mse_map)

    return orr_list, orr_mse_list
