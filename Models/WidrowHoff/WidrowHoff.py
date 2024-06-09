"""
Widrow-Hoff (LMS) Learning Script

This script contains functions for implementing the Widrow-Hoff learning algorithm and related analysis for linear
regression tasks.
It utilizes scikit-learn for K-Fold cross-validation and various utility functions for measures, predictions, plotting,
and constant definitions.

Functions:
- `widrow_hoff`: Implement the Widrow-Hoff learning algorithm for optimizing linear regression coefficients.
- `widrow_hoff_KFold`: Perform K-Fold cross-validation with the Widrow-Hoff algorithm for linear regression.
- `widrow_hoff_adversarial`: Evaluate the Widrow-Hoff algorithm on adversarial test data.
- `widrow_hoff_convergence`: Perform Widrow-Hoff algorithm for linear regression with convergence analysis.
- `widrow_hoff_plot_convergence`: Perform Widrow-Hoff algorithm with convergence analysis and plot results.

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


def widrow_hoff(X, y, learning_rate, modular):
    """
        Implement the Widrow-Hoff learning algorithm for optimizing linear regression coefficients.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            learning_rate (float): Learning rate for the algorithm.

        Returns:
            w (array): Optimized coefficient vector.
            epoch_list (array): List of epoch indices.
            cost_list (array): List of corresponding costs.
        """
    cost_list = np.array([])
    epoch_list = np.array([])

    n_samples, n_features = X.shape
    w = np.zeros(n_features + 1)
    x0 = np.ones(len(X))
    X = np.concatenate((np.matrix(x0).T, X), axis=1)
    i = 0
    for xs, ys in zip(X, y):
        xs = np.squeeze(np.asarray(xs))
        w = w - (2 * learning_rate * (((np.dot(w.T, xs)) - ys) * xs))

        y_predicted = np.dot(w, xs)
        cost = np.square(ys - y_predicted)  # make sure this is correct cost.

        if i % modular == 0:  # at every 50 iteration record the cost and epoch value
            cost_list = np.append(cost_list, cost)
            epoch_list = np.append(epoch_list, i)

        i += 1

    return w, epoch_list, cost_list


def widrow_hoff_KFold(X, y, learning_rate, seed, modular):
    """
        Perform K-Fold cross-validation with the Widrow-Hoff algorithm for linear regression.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            learning_rate (float): Learning rate for the algorithm.
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
        w, epoch_list, cost_list = widrow_hoff(X_train, y_train, learning_rate, modular)
        y_predicted = Predictions._compute_predictions_(X_test, w)
        acc_per_split_for_same_seed = Measures.r2_score_(y_test, y_predicted)
        scores.append(acc_per_split_for_same_seed)
        avg_cost = np.array(cost_list).mean()
        costs.append(avg_cost)

    return np.array(scores).mean(), np.array(costs).mean()


def widrow_hoff_adversarial(X_train, y_train, X_test, y_test, learning_rate):
    """
        Evaluate the Widrow-Hoff algorithm on adversarial test data.

        Args:
            X_train (array-like): Training input feature matrix.
            y_train (array-like): Training target values.
            X_test (array-like): Adversarial test input feature matrix.
            y_test (array-like): Adversarial test target values.
            learning_rate (float): Learning rate for the algorithm.

        Returns:
            acc (float): Accuracy (R-squared) on adversarial test data.
        """
    w, epoch_list, cost_list = widrow_hoff(X_train, y_train, learning_rate)
    y_predicted = Predictions._compute_predictions_(X_test, w)
    acc = Measures.r2_score_(y_test, y_predicted)
    return acc


def widrow_hoff_convergence(X, y, learning_rate):
    """
        Perform Widrow-Hoff algorithm for linear regression with convergence analysis.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            learning_rate (float): Learning rate for the algorithm.

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
        w, epoch_list, cost_list = widrow_hoff(X_train, y_train, learning_rate)
        epoch_list_per_seed = Util.sum_lists_element_wise(epoch_list_per_seed, epoch_list)
        cost_list_per_seed = Util.sum_lists_element_wise(cost_list_per_seed, cost_list)
        y_predicted = Predictions._compute_predictions_(X_test, w)
        acc_per_split_for_same_seed = Measures.r2_score_(y_test, y_predicted)
        scores.append(acc_per_split_for_same_seed)
    acc = np.array(scores).mean()
    epochs_accu = epoch_list_per_seed / n_splits
    cost_accu = cost_list_per_seed / n_splits
    return acc, epochs_accu, cost_accu


def widrow_hoff_plot_convergence(X, y, learning_rate, X_test, y_test, model_name):
    """
        Perform Widrow-Hoff algorithm with convergence analysis and plot results.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            learning_rate (float): Learning rate for the algorithm.
            X_test (array-like): Test input feature matrix.
            y_test (array-like): Test target values.
            model_name (str): Name of the model for plotting purposes.

        Returns:
            w (array): Optimized coefficient vector.
        """
    Util.create_directory(Constants.plotting_path + model_name)
    n_samples, n_features = X.shape
    w = np.zeros(n_features + 1)
    x0 = np.ones(len(X))
    X_augmented = np.concatenate((np.matrix(x0).T, X), axis=1)

    accumulated_xs = []  # this will append each single xs on each iteration for plotting reasons
    accumulated_ys = []  # this will append each single ys on each iteration for plotting reasons
    for iteration, (xs, ys) in enumerate(zip(X_augmented, y)):
        xs = np.squeeze(np.asarray(xs))
        w = w - (2 * learning_rate * (((np.dot(w.T, xs)) - ys) * xs))

        # enable plotting at each iteration (only for 2D)
        accumulated_xs.append(xs)
        accumulated_ys.append(ys)

        Plotter.compute_acc_plot_per_iteration(X_train=X, y_train=y, w=w, b=None,
                                               iteration=iteration, X_test=X_test, y_test=y_test,
                                               accumulated_xs=accumulated_xs, accumulated_ys=accumulated_ys,
                                               model_name=model_name)
    return w




def widrow_hoff_plot_convergence2(X, y, learning_rate, X_test, y_test, model_name, modular):
    """
        Perform Widrow-Hoff algorithm with convergence analysis and plot results.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            learning_rate (float): Learning rate for the algorithm.
            X_test (array-like): Test input feature matrix.
            y_test (array-like): Test target values.
            model_name (str): Name of the model for plotting purposes.

        Returns:
            w (array): Optimized coefficient vector.
        """
    n_samples, n_features = X.shape
    w = np.zeros(n_features + 1)
    x0 = np.ones(len(X))
    X_augmented = np.concatenate((np.matrix(x0).T, X), axis=1)
    widrow_hoff_map = {}
    widrow_hoff_mse_map = {}

    accumulated_size = 0
    for iteration, (xs, ys) in enumerate(zip(X_augmented, y)):
        xs = np.squeeze(np.asarray(xs))
        w = w - (2 * learning_rate * (((np.dot(w.T, xs)) - ys) * xs))
        accumulated_size +=1

        if(accumulated_size % modular ==0):
            y_predicted = Predictions._compute_predictions_(X_test, w)
            acc = Measures.r2_score_(y_test, y_predicted)
            widrow_hoff_map[accumulated_size] = acc

            mse = np.mean((y_test - y_predicted)**2)
            widrow_hoff_mse_map[accumulated_size] = mse



    return w, widrow_hoff_map, widrow_hoff_mse_map

def widrow_hoff_convergence2(X, y, learning_rate, model_name, seed, modular):

    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    widrow_hoff_list = []
    widrow_hoff_mse_list = []
    for fold_index, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, widrow_hoff_map, widrow_hoff_mse_map = widrow_hoff_plot_convergence2(X_train, y_train, learning_rate, X_test,
                                                                      y_test, model_name, modular)
        widrow_hoff_list.append(widrow_hoff_map)
        widrow_hoff_mse_list.append(widrow_hoff_mse_map)

    return widrow_hoff_list, widrow_hoff_mse_list
