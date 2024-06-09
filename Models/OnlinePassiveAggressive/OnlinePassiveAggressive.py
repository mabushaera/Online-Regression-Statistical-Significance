"""
Online Passive-Aggressive Regression Script

This script contains functions for performing Online Passive-Aggressive Regression.
It utilizes scikit-learn for K-Fold cross-validation and various utility functions for measures, predictions,
plotting, and constant definitions.

Functions:
- `online_passive_aggressive`: Implement the Online Passive-Aggressive Regression algorithm for linear regression.
- `online_passive_aggressive_KFold`: Perform K-Fold cross-validation with Online Passive-Aggressive Regression.
- `online_passive_aggressive_adversarial`: Evaluate Online Passive-Aggressive Regression on adversarial test data.
- `online_passive_aggressive_convergence`: Perform Online Passive-Aggressive Regression with convergence analysis.
- `online_passive_aggressive_plot_convergence`: Perform Online Passive-Aggressive Regression with convergence analysis
    and plot results.

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


def online_passive_aggressive(X, y, C, epsilon, modular):
    """
        Implement the Online Passive-Aggressive Regression algorithm for linear regression.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            C (float): Aggressiveness parameter controlling update step size.
            epsilon (float): Epsilon parameter determining sensitivity to prediction errors.

        Returns:
            w (array): Optimized coefficient vector.
            epoch_list (array): List of epoch indices.
            cost_list (array): List of corresponding costs.

        """

    n_samples, n_features = X.shape
    w = np.zeros(n_features)

    cost_list = np.array([])
    epoch_list = np.array([])

    for i in range(n_samples):
        x = X[i]
        y_true = y[i]

        y_pred = np.dot(w, x)

        # epsilon_insensitive_hinge_loss
        loss = max(0, abs(y_pred - y_true) - epsilon)

        # Calculate lagrange multiplier T
        # T = loss / (np.linalg.norm(x) ** 2)  # for PA1
        # T = min(C, (loss / (np.linalg.norm(x) ** 2)))  # for PA2
        T = loss / (np.linalg.norm(x) ** 2 + 1 / (2 * C))  # for PA3

        # Update weights
        w += T * np.sign(y_true - y_pred) * x

        cost = np.square(y_true - y_pred)  # make sure this is correct cost.

        if i % modular == 0:  # at every 50 iteration record the cost and epoch value
            cost_list = np.append(cost_list, cost)
            epoch_list = np.append(epoch_list, i)

    return w, epoch_list, cost_list


def online_passive_aggressive_KFold(X, y, C, epsilon, seed, modular):
    """
        Perform K-Fold cross-validation with Online Passive-Aggressive Regression.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            C (float): Aggressiveness parameter controlling update step size.
            epsilon (float): Epsilon parameter determining sensitivity to prediction errors.
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
        w, epoch_list, cost_list = online_passive_aggressive(X_train, y_train,
                                                             C, epsilon, modular)
        predicted_y_test = Predictions.compute_predictions(X_test, w)
        acc_per_split_for_same_seed = Measures.r2_score_(y_test, predicted_y_test)
        scores.append(acc_per_split_for_same_seed)
        avg_cost = np.array(cost_list).mean()
        costs.append(avg_cost)


    return np.array(scores).mean(),np.array(costs).mean()


def online_passive_aggressive_adversarial(X_train, y_train, X_test, y_test, C, epsilon):
    """
        Evaluate Online Passive-Aggressive Regression on adversarial test data.

        Args:
            X_train (array-like): Training input feature matrix.
            y_train (array-like): Training target values.
            X_test (array-like): Adversarial test input feature matrix.
            y_test (array-like): Adversarial test target values.
            C (float): Aggressiveness parameter controlling update step size.
            epsilon (float): Epsilon parameter determining sensitivity to prediction errors.

        Returns:
            acc (float): Accuracy (R-squared) on adversarial test data.

        """
    w, epoch_list, cost_list = online_passive_aggressive(X_train, y_train, C, epsilon)
    predicted_y_test = Predictions.compute_predictions(X_test, w)
    acc = Measures.r2_score_(y_test, predicted_y_test)
    return acc


def online_passive_aggressive_convergence(X, y, regularization_parameter, epsilon):
    """
        Perform Online Passive-Aggressive Regression with convergence analysis.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            C (float): Aggressiveness parameter controlling update step size.
            epsilon (float): Epsilon parameter determining sensitivity to prediction errors.

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
        w, epoch_list, cost_list = online_passive_aggressive(X_train, y_train, regularization_parameter, epsilon)
        epoch_list_per_seed = Util.sum_lists_element_wise(epoch_list_per_seed, epoch_list)
        cost_list_per_seed = Util.sum_lists_element_wise(cost_list_per_seed, cost_list)
        predicted_y_test = Predictions.compute_predictions(X_test, w)
        acc_per_split_for_same_seed = Measures.r2_score_(y_test, predicted_y_test)
        scores.append(acc_per_split_for_same_seed)

    acc = np.array(scores).mean()
    epochs_accu = epoch_list_per_seed / n_splits
    cost_accu = cost_list_per_seed / n_splits
    return acc, epochs_accu, cost_accu


def online_passive_aggressive_plot_convergence(X, y, C, epsilon, X_test, y_test, model_name):
    """
        Perform Online Passive-Aggressive Regression with convergence analysis and plot results.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            C (float): Aggressiveness parameter controlling update step size.
            epsilon (float): Epsilon parameter determining sensitivity to prediction errors.
            X_test (array-like): Test input feature matrix.
            y_test (array-like): Test target values.
            model_name (str): Name of the model for plotting purposes.

        Returns:
            w (array): Optimized coefficient vector.

        """
    Util.create_directory(Constants.plotting_path + model_name)
    n_samples, n_features = X.shape
    w = np.zeros(n_features)

    accumulated_xs = np.array([])
    accumulated_ys = np.array([])

    for i in range(n_samples):
        x = X[i]
        y_true = y[i]

        accumulated_xs = np.append(accumulated_xs, x)
        accumulated_ys = np.append(accumulated_ys, y_true)

        y_pred = np.dot(w, x)

        # epsilon_insensitive_hinge_loss
        loss = max(0, abs(y_pred - y_true) - epsilon)

        # Calculate lagrange multiplier T
        # T = loss / (np.linalg.norm(x) ** 2)  # for PA1
        # T = min(self.C, (loss / (np.linalg.norm(x) ** 2)))  # for PA2
        T = loss / (np.linalg.norm(x) ** 2 + 1 / (2 * C))  # for PA3

        # Update weights
        w += T * np.sign(y_true - y_pred) * x

        Plotter.compute_acc_plot_per_iteration(X_train=X, y_train=y, w=w, b=None,
                                               iteration=i, X_test=X_test, y_test=y_test,
                                               accumulated_xs=accumulated_xs, accumulated_ys=accumulated_ys,
                                               model_name=model_name)

    return w


def online_passive_aggressive_plot_convergence2(X, y, C, epsilon, X_test, y_test, model_name, modular):
    """
        Perform Online Passive-Aggressive Regression with convergence analysis and plot results.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            C (float): Aggressiveness parameter controlling update step size.
            epsilon (float): Epsilon parameter determining sensitivity to prediction errors.
            X_test (array-like): Test input feature matrix.
            y_test (array-like): Test target values.
            model_name (str): Name of the model for plotting purposes.

        Returns:
            w (array): Optimized coefficient vector.

        """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    accumulated_size = 0
    pa_map = {}
    pa_mse_map = {}
    for i in range(n_samples):
        x = X[i]
        y_true = y[i]

        accumulated_size +=1

        y_pred = np.dot(w, x)

        # epsilon_insensitive_hinge_loss
        loss = max(0, abs(y_pred - y_true) - epsilon)

        # Calculate lagrange multiplier T
        # T = loss / (np.linalg.norm(x) ** 2)  # for PA1
        # T = min(self.C, (loss / (np.linalg.norm(x) ** 2)))  # for PA2
        T = loss / (np.linalg.norm(x) ** 2 + 1 / (2 * C))  # for PA3

        # Update weights
        w += T * np.sign(y_true - y_pred) * x

        if accumulated_size % modular == 0:
            y_predicted = Predictions.compute_predictions(X_test, w)
            acc = Measures.r2_score_(y_test, y_predicted)
            pa_map[accumulated_size] = acc

            mse = np.mean((y_test - y_predicted)**2)
            pa_mse_map[accumulated_size] = mse


    return w, pa_map, pa_mse_map

def online_passive_aggressive_convergence2(X, y, C, epsilon, model_name, seed, modular):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    pa_list = []
    pa_mse_list = []
    for fold_index, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, pa_map, pa_mse_map = online_passive_aggressive_plot_convergence2(X, y, C, epsilon, X_test, y_test, model_name, modular)
        pa_list.append(pa_map)
        pa_mse_list.append(pa_mse_map)

    return pa_list, pa_mse_list