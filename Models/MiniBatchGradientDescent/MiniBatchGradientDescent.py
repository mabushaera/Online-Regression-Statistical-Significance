"""
Mini-Batch Gradient Descent Script

This script contains functions for performing mini-batch gradient descent and related analysis on linear regression
models.
It leverages tools from the scikit-learn library and custom utility functions for various tasks such as performance
evaluation, plotting, and data manipulation.

Functions:
- `mini_batch_gradient_descent`: Perform mini-batch gradient descent to optimize linear regression coefficients.
- `mini_batch_gradient_descent_KFold`: Perform K-Fold cross-validation with mini-batch gradient descent for linear
   regression.
- `mini_batch_gradient_descent_adversarial`: Perform mini-batch gradient descent for linear regression and evaluate on
   adversarial test data.
- `mini_batch_gradient_descent_convergence`: Perform mini-batch gradient descent for linear regression with convergence
analysis.
- `mini_batch_stochastic_gradient_descent_plot_convergence`: Perform mini-batch stochastic gradient descent with
   convergence analysis and plot results.

Dependencies:
- `numpy`: Numerical library for array manipulation and calculations.
- `sklearn.model_selection.KFold`: K-Fold cross-validation for splitting data into training and test sets.
- `Utils`: Custom utility module providing measures, predictions, plotting, and constant definitions.
- `Measures`: Utility functions for evaluating performance measures like R-squared.
- `Predictions`: Utility functions for computing predictions using model coefficients.
- `Util`: Utility functions for data manipulation and analysis.
- `Plotter`: Utility functions for creating plots and visualizing data.
- `Constants`: Module containing constant definitions for paths, models names and configurations.

Author: M. Shaira
Date: Aug, 2023.
"""

import numpy as np
from sklearn.model_selection import KFold
from Utils import Measures, Predictions, Util, Plotter, Constants


def mini_batch_gradient_descent(X, y, epochs=100, batch_size=10, learning_rate=0.01, modular=None):
    """
        Perform mini-batch gradient descent to optimize linear regression coefficients.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            epochs (int): Number of training epochs (default: 100).
            batch_size (int): Size of mini-batches (default: 5).
            learning_rate (float): Learning rate for gradient descent (default: 0.01).

        Returns:
            w (array): Optimized coefficient vector.
            b (float): Optimized intercept.
            epoch_list (list): List of epoch indices.
            cost_list (list): List of corresponding costs (MSE).

        """

    number_of_features = X.shape[1]
    w = np.zeros(shape=number_of_features)
    b = 0
    total_samples = X.shape[0]  # number of rows in X

    if batch_size > total_samples:  # In this case mini-batch becomes the same as batch gradient descent
        batch_size = total_samples

    cost_list = []
    epoch_list = []

    accumulative_size = 0

    for i in range(epochs):
        batch_indices = np.random.choice(total_samples, batch_size, replace=False)  # Randomly select batch_size indices

        X_batch = X[batch_indices]
        y_batch = y[batch_indices]

        accumulative_size += X_batch.shape[0]

        y_predicted = np.dot(w, X_batch.T) + b

        w_grad = -(2 / batch_size) * (X_batch.T.dot(y_batch - y_predicted))
        b_grad = -(2 / batch_size) * np.sum(y_batch - y_predicted)

        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad

        cost = np.mean(np.square(y_batch - y_predicted))  # MSE (Mean Squared Error)

        # if i % modular == 0:
        cost_list.append(cost)
        epoch_list.append(accumulative_size)

    return w, b, epoch_list, cost_list


def mini_batch_gradient_descent_KFold(X, y, epochs, batch_size, learning_rate, seed, modular):
    """
        Perform K-Fold cross-validation with mini-batch gradient descent for linear regression.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            epochs (int): Number of training epochs.
            batch_size (int): Size of mini-batches.
            learning_rate (float): Learning rate for gradient descent.
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
        w, b, epoch_list, cost_list = mini_batch_gradient_descent(X_train, y_train, epochs,
                                                                  batch_size, learning_rate, modular)
        y_predicted = Predictions.compute_predictions_(X_test, w, b)
        acc_per_split_for_same_seed = Measures.r2_score_(y_test, y_predicted)
        scores.append(acc_per_split_for_same_seed)
        avg_cost = np.array(cost_list).mean()
        costs.append(avg_cost)

    return np.array(scores).mean(), np.array(costs).mean()


def mini_batch_gradient_descent_adversarial(X_train, y_train, X_test, y_test, epochs, batch_size, learning_rate):
    """
        Perform mini-batch gradient descent for linear regression and evaluate on adversarial test data.

        Args:
            X_train (array-like): Training input feature matrix.
            y_train (array-like): Training target values.
            X_test (array-like): Adversarial test input feature matrix.
            y_test (array-like): Adversarial test target values.
            epochs (int): Number of training epochs.
            batch_size (int): Size of mini-batches.
            learning_rate (float): Learning rate for gradient descent.

        Returns:
            acc (float): Accuracy (R-squared) on adversarial test data.

        """
    w, b, epoch_list, cost_list = mini_batch_gradient_descent(X_train, y_train, epochs,
                                                              batch_size, learning_rate)
    y_predicted = Predictions.compute_predictions_(X_test, w, b)
    acc = Measures.r2_score_(y_test, y_predicted)
    return acc


def mini_batch_gradient_descent_convergence(X, y, epochs, batch_size, learning_rate):
    """
        Perform mini-batch gradient descent for linear regression for convergence analysis.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            epochs (int): Number of training epochs.
            batch_size (int): Size of mini-batches.
            learning_rate (float): Learning rate for gradient descent.

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
        w, b, epoch_list, cost_list = mini_batch_gradient_descent(X_train, y_train, epochs,
                                                                  batch_size, learning_rate)
        epoch_list_per_seed = Util.sum_lists_element_wise(epoch_list_per_seed, epoch_list)
        cost_list_per_seed = Util.sum_lists_element_wise(cost_list_per_seed, cost_list)
        y_predicted = Predictions.compute_predictions_(X_test, w, b)
        acc_per_split_for_same_seed = Measures.r2_score_(y_test, y_predicted)
        scores.append(acc_per_split_for_same_seed)
    acc = np.array(scores).mean()
    epochs_accu = epoch_list_per_seed / n_splits
    cost_accu = cost_list_per_seed / n_splits
    return acc, epochs_accu, cost_accu


def mini_batch_stochastic_gradient_descent_plot_convergence(X, y, epochs, batch_size, learning_rate, X_test, y_test,
                                                            model_name):
    """
        Perform mini-batch stochastic gradient descent with convergence analysis and plot results.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            epochs (int): Number of training epochs.
            batch_size (int): Size of mini-batches.
            learning_rate (float): Learning rate for gradient descent.
            X_test (array-like): Test input feature matrix.
            y_test (array-like): Test target values.
            model_name (str): Name of the model for plotting purposes.

        Returns:
            w (array): Optimized coefficient vector.
            b (float): Optimized intercept.

        """
    n_features = X.shape[1]
    Util.create_directory(Constants.plotting_path + model_name)
    w = np.zeros(shape=n_features)
    b = 0
    total_samples = X.shape[0]  # number of rows in X
    batch_size = 10 # max(batch_size, (n_features + 1) * 5)
    if batch_size > total_samples:  # In this case mini batch becomes same as batch gradient descent
        batch_size = total_samples

    accumulated_xs = []  # this will append each single xs on each iteration for plotting reasons
    accumulated_ys = []  # this will append each single ys on each iteration for plotting reasons
    for i in range(epochs):
        batch_indices = np.random.choice(total_samples, batch_size, replace=False)  # Randomly select batch_size indices
        Xj = X[batch_indices]
        yj = y[batch_indices]

        accumulated_xs = np.concatenate((np.array(accumulated_xs), np.array(Xj).flatten()))
        accumulated_ys = np.concatenate((np.array(accumulated_ys), np.array(yj)))

        y_predicted = np.dot(w, Xj.T) + b

        w_grad = -(2 / batch_size) * (Xj.T.dot(yj - y_predicted))
        b_grad = -(2 / batch_size) * np.sum(yj - y_predicted)

        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad

        Plotter.compute_acc_plot_per_iteration(X_train=X, y_train=y, w=w, b=b,
                                               iteration=len(accumulated_xs), X_test=X_test, y_test=y_test,
                                               accumulated_xs=accumulated_xs, accumulated_ys=accumulated_ys,
                                               model_name=model_name)

    return w, b




def mini_batch_stochastic_gradient_descent_plot_convergence2(X, y, epochs, batch_size, learning_rate, X_test, y_test, model_name):
    """
        Perform mini-batch stochastic gradient descent with convergence analysis and plot results.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            epochs (int): Number of training epochs.
            batch_size (int): Size of mini-batches.
            learning_rate (float): Learning rate for gradient descent.
            model_name (str): Name of the model for plotting purposes.

        Returns:
            w (array): Optimized coefficient vector.
            b (float): Optimized intercept.

        """
    n_features = X.shape[1]
    w = np.zeros(shape=n_features)
    b = 0
    total_samples = X.shape[0]  # number of rows in X

    if batch_size > total_samples:  # In this case mini batch becomes same as batch gradient descent
        batch_size = total_samples

    # accumulated_xs = []  # this will append each single xs on each iteration for plotting reasons
    # accumulated_ys = []  # this will append each single ys on each iteration for plotting reasons
    accumulated_size = 0;
    mbgd_map = {}
    mbgd_mse_map = {}
    for i in range(epochs):
        batch_indices = np.random.choice(total_samples, batch_size, replace=False)  # Randomly select batch_size indices
        Xj = X[batch_indices]
        yj = y[batch_indices]

        # accumulated_xs = np.concatenate((np.array(accumulated_xs), np.array(Xj).flatten()))
        # accumulated_ys = np.concatenate((np.array(accumulated_ys), np.array(yj)))
        accumulated_size += Xj.shape[0]

        y_predicted = np.dot(w, Xj.T) + b

        w_grad = -(2 / batch_size) * (Xj.T.dot(yj - y_predicted))
        b_grad = -(2 / batch_size) * np.sum(yj - y_predicted)

        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad

        # if accumulated_size % 10 == 0:
        y_predicted = Predictions.compute_predictions_(X_test, w, b)
        acc = Measures.r2_score_(y_test, y_predicted)
        mbgd_map[accumulated_size] = "{:.5f}".format(acc)

        mse = np.mean((y_test - y_predicted)**2)
        mbgd_mse_map[accumulated_size] = "{:.5f}".format(mse)

    return w, b, mbgd_map, mbgd_mse_map


def mini_batch_gradient_descent_convergence2(X, y, epochs, batch_size, learning_rate, model_name, seed):
    n_splits = 5
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    sgd_list = []
    sgd_mse_list = []
    for fold_index, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, b, sgd_map, sgd_mse_map = mini_batch_stochastic_gradient_descent_plot_convergence2(X_train, y_train, epochs, batch_size, learning_rate, X_test, y_test, model_name)
        sgd_list.append(sgd_map)
        sgd_mse_list.append(sgd_mse_map)

    return sgd_list, sgd_mse_list

