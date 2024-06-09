"""
Online Lasso Regression Script

This script contains functions for performing Online Lasso Regression, a linear regression technique that adds L1
regularization
to the standard regression loss to promote sparsity in the model weights. It utilizes scikit-learn for K-Fold
cross-validation
and various utility functions for measures, predictions, plotting, and constant definitions.

Functions:
- `online_lasso_regression`: Implement the Online Lasso Regression algorithm for linear regression.
- `online_lasso_regression_KFold`: Perform K-Fold cross-validation with Online Lasso Regression.
- `online_lasso_regression_adversarial`: Evaluate Online Lasso Regression on adversarial test data.
- `online_lasso_regression_convergence`: Perform Online Lasso Regression with convergence analysis.
- `online_lasso_regression_plot_convergence`: Perform Online Lasso Regression with convergence analysis and plot results.

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


def online_lasso_regression(X, y, learning_rate, epochs, regularization_param, modular):
    """
        Implement the Online Lasso Regression algorithm for linear regression.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            learning_rate (float): Learning rate controlling the update step size.
            epochs (int): Number of iterations (epochs).
            regularization_param (float): L1 regularization parameter.

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

        dw = 2 * x_sample * (y_predicted - y_sample) + regularization_param * np.sign(w)
        db = 2 * (y_predicted - y_sample)

        w -= learning_rate * dw
        bias -= learning_rate * db

        cost = np.square(y_sample - y_predicted)

        if i % modular == 0:  # at every 50 iteration record the cost and epoch value
            cost_list = np.append(cost_list, cost)
            epoch_list = np.append(epoch_list, i)

    return w, bias, epoch_list, cost_list


def online_lasso_regression_KFold(X, y, learning_rate, epochs, regularization_param, seed, modular):
    """
        Perform K-Fold cross-validation with Online Lasso Regression.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            learning_rate (float): Learning rate controlling the update step size.
            epochs (int): Number of iterations (epochs).
            regularization_param (float): L1 regularization parameter.
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
        w, b, epoch_list, cost_list = online_lasso_regression(X_train, y_train, learning_rate,
                                                              epochs, regularization_param, modular)
        predicted_y_test = Predictions.compute_predictions_(X_test, w, b)
        acc_per_split_for_same_seed = Measures.r2_score_(y_test, predicted_y_test)
        scores.append(acc_per_split_for_same_seed)
        avg_cost = np.array(cost_list).mean()
        costs.append(avg_cost)


    return np.array(scores).mean(), np.array(costs).mean()


def online_lasso_regression_adversarial(X_train, y_train, X_test, y_test, learning_rate, epochs, regularization_param):
    """
    Evaluate Online Lasso Regression on adversarial test data.

    Args:
        X_train (array-like): Training input feature matrix.
        y_train (array-like): Training target values.
        X_test (array-like): Adversarial test input feature matrix.
        y_test (array-like): Adversarial test target values.
        learning_rate (float): Learning rate controlling the update step size.
        epochs (int): Number of iterations (epochs).
        regularization_param (float): L1 regularization parameter.

    Returns:
        acc (float): Accuracy (R-squared) on adversarial test data.

    """
    w, b, epoch_list, cost_list = online_lasso_regression(X_train, y_train, learning_rate, epochs,
                                                          regularization_param)
    predicted_y_test = Predictions.compute_predictions_(X_test, w, b)
    acc = Measures.r2_score_(y_test, predicted_y_test)

    return acc


def online_lasso_regression_convergence(X, y, learning_rate, epochs, regularization_param):
    """
        Perform Online Lasso Regression with convergence analysis.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            learning_rate (float): Learning rate controlling the update step size.
            epochs (int): Number of iterations (epochs).
            regularization_param (float): L1 regularization parameter.

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
        w, b, epoch_list, cost_list = online_lasso_regression(X_train, y_train, learning_rate,
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


def online_lasso_regression_plot_convergence(X, y, learning_rate, epochs, regularization_param, X_test, y_test,
                                             model_name):
    """
        Perform Online Lasso Regression with convergence analysis and plot results.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            learning_rate (float): Learning rate controlling the update step size.
            epochs (int): Number of iterations (epochs).
            regularization_param (float): L1 regularization parameter.
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

        dw = 2 * x_sample * (y_predicted - y_sample) + regularization_param * np.sign(w)
        db = 2 * (y_predicted - y_sample)

        w -= learning_rate * dw
        bias -= learning_rate * db

        Plotter.compute_acc_plot_per_iteration(X_train=X, y_train=y, w=w, b=bias,
                                               iteration=i, X_test=X_test, y_test=y_test,
                                               accumulated_xs=accumulated_xs, accumulated_ys=accumulated_ys,
                                               model_name=model_name)

    return w, bias



def online_lasso_regression_plot_convergence2(X, y, learning_rate, epochs, regularization_param, X_test, y_test,
                                             model_name, modular):
    """
        Perform Online Lasso Regression with convergence analysis and plot results.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            learning_rate (float): Learning rate controlling the update step size.
            epochs (int): Number of iterations (epochs).
            regularization_param (float): L1 regularization parameter.
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
    olr_map = {}
    olr_mse_map = {}
    for i in range(epochs):
        index = np.random.randint(num_of_samples)
        x_sample = X[index]
        y_sample = y[index]
        accumulated_size +=1

        y_predicted = np.dot(x_sample, w.T) + bias

        dw = 2 * x_sample * (y_predicted - y_sample) + regularization_param * np.sign(w)
        db = 2 * (y_predicted - y_sample)

        w -= learning_rate * dw
        bias -= learning_rate * db

        if(accumulated_size %modular ==0):
            y_predicted = Predictions.compute_predictions_(X_test, w, bias)
            acc = Measures.r2_score_(y_test, y_predicted)
            olr_map[accumulated_size] = acc

            mse = np.mean((y_test - y_predicted)**2)
            olr_mse_map[accumulated_size] = mse


    return w, bias, olr_map, olr_mse_map


def online_lasso_regression_convergence2(X, y, learning_rate, epochs, regularization_param, model_name, seed, modular):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    olr_list = []
    olr_mse_list = []
    for fold_index, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, b, olr_map, olr_mse_map = online_lasso_regression_plot_convergence2(X, y, learning_rate, epochs, regularization_param, X_test, y_test, model_name, modular)
        olr_list.append(olr_map)
        olr_mse_list.append(olr_mse_map)

    return olr_list, olr_mse_list