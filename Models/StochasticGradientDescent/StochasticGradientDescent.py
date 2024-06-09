"""
Stochastic Gradient Descent Script

This script contains functions for implementing the Stochastic Gradient Descent (SGD) algorithm and related analysis
for linear regression tasks. It utilizes scikit-learn for K-Fold cross-validation and various utility functions for
measures,
predictions, plotting, and constant definitions.

Functions:
- `stochastic_gradient_descent`: Implement the Stochastic Gradient Descent algorithm for optimizing linear regression
    coefficients.
- `stochastic_gradient_descent_KFold`: Perform K-Fold cross-validation with Stochastic Gradient Descent for linear
    regression.
- `stochastic_gradient_descent_adversarial`: Evaluate the Stochastic Gradient Descent algorithm on adversarial test
    data.
- `stochastic_gradient_descent_convergence`: Perform Stochastic Gradient Descent for linear regression with convergence
    analysis.
- `stochastic_gradient_descent_plot_convergence`: Perform Stochastic Gradient Descent with convergence analysis and plot
    results.

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

def stochastic_gradient_descent(X, y_true, epochs, learning_rate=0.01, modular=None):
    """
        Implement the Stochastic Gradient Descent algorithm for optimizing linear regression coefficients.

        Args:
            X (array-like): Input feature matrix.
            y_true (array-like): True target values.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the algorithm.

        Returns:
            w (array): Optimized coefficient vector.
            b (float): Optimized intercept.
            epoch_list (array): List of epoch indices.
            cost_list (array): List of corresponding costs.

        """
    total_samples, number_of_features = X.shape
    w = np.zeros(shape=number_of_features)
    b = 0

    cost_list = np.array([])
    epoch_list = np.array([])



    for i in range(epochs):
        random_index = np.random.randint(total_samples)

        sample_x = X[random_index]
        sample_y = y_true[random_index]

        y_predicted = np.dot(w, sample_x.T) + b

        w_grad = 2 * sample_x * (y_predicted - sample_y)
        b_grad = 2 * (y_predicted - sample_y)

        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad

        cost = np.square(sample_y - y_predicted)

        if i % modular == 0:  # at every 50 iteration record the cost and epoch value
            cost_list = np.append(cost_list, cost)
            epoch_list = np.append(epoch_list, i)

    return w, b, epoch_list, cost_list


def stochastic_gradient_descent_KFold(X, y, epochs, learning_rate, seed, modular):
    """
        Perform K-Fold cross-validation with Stochastic Gradient Descent for linear regression.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            epochs (int): Number of training epochs.
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
        w, b, epoch_list, cost_list = stochastic_gradient_descent(X_train, y_train, epochs,
                                                                  learning_rate, modular)
        y_predicted = Predictions.compute_predictions_(X_test, w, b)
        acc_per_split_for_same_seed = Measures.r2_score_(y_test, y_predicted)
        scores.append(acc_per_split_for_same_seed)
        avg_cost = np.array(cost_list).mean()
        costs.append(avg_cost)


    return np.array(scores).mean(), np.array(costs).mean()


def stochastic_gradient_descent_adversarial(X_train, y_train, X_test, y_test, epochs, learning_rate):
    """
        Evaluate the Stochastic Gradient Descent algorithm on adversarial test data.

        Args:
            X_train (array-like): Training input feature matrix.
            y_train (array-like): Training target values.
            X_test (array-like): Adversarial test input feature matrix.
            y_test (array-like): Adversarial test target values.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the algorithm.

        Returns:
            acc (float): Accuracy (R-squared) on adversarial test data.

        """
    w, b, epoch_list, cost_list = stochastic_gradient_descent(X_train, y_train, epochs, learning_rate)
    y_predicted = Predictions.compute_predictions_(X_test, w, b)
    acc = Measures.r2_score_(y_test, y_predicted)

    return acc


def stochastic_gradient_descent_convergence(X, y, epochs, learning_rate):
    """
        Perform Stochastic Gradient Descent for linear regression with convergence analysis.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            epochs (int): Number of training epochs.
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
    for fold_index, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, b, epoch_list, cost_list = stochastic_gradient_descent(X_train, y_train, epochs, learning_rate)
        epoch_list_per_seed = Util.sum_lists_element_wise(epoch_list_per_seed, epoch_list)
        cost_list_per_seed = Util.sum_lists_element_wise(cost_list_per_seed, cost_list)
        y_predicted = Predictions.compute_predictions_(X_test, w, b)
        acc_per_split_for_same_seed = Measures.r2_score_(y_test, y_predicted)
        scores.append(acc_per_split_for_same_seed)

    acc = np.array(scores).mean()
    epochs_accu = epoch_list_per_seed / n_splits
    cost_accu = cost_list_per_seed / n_splits

    return acc, epochs_accu, cost_accu


def stochastic_gradient_descent_plot_convergence(X, y, epochs, learning_rate, X_test, y_test, model_name):
    """
        Perform Stochastic Gradient Descent with convergence analysis and plot results.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the algorithm.
            X_test (array-like): Test input feature matrix.
            y_test (array-like): Test target values.
            model_name (str): Name of the model for plotting purposes.

        Returns:
            w (array): Optimized coefficient vector.
            b (float): Optimized intercept.

        """
    Util.create_directory(Constants.plotting_path + model_name)
    n_features = X.shape[1]
    w = np.zeros(shape=n_features)
    b = 0
    total_samples = X.shape[0]

    accumulated_xs = []  # this will append each single xs on each iteration for plotting reasons
    accumulated_ys = []  # this will append each single ys on each iteration for plotting reasons

    for i in range(epochs):
        random_index = np.random.randint(total_samples)

        sample_x = X[random_index]
        sample_y = y[random_index]

        accumulated_xs.append(sample_x)
        accumulated_ys.append(sample_y)

        y_predicted = np.dot(w, sample_x.T) + b

        w_grad = 2 * sample_x * (y_predicted - sample_y)
        b_grad = 2 * (y_predicted - sample_y)

        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad

        Plotter.compute_acc_plot_per_iteration(X_train=X, y_train=y, w=w, b=b, iteration=i, X_test=X_test,
                                               y_test=y_test,
                                               accumulated_xs=accumulated_xs, accumulated_ys=accumulated_ys,
                                               model_name=model_name)

    return w, b



def stochastic_gradient_descent_plot_convergence(X, y, epochs, learning_rate, X_test, y_test, model_name):
    """
        Perform Stochastic Gradient Descent with convergence analysis and plot results.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the algorithm.
            X_test (array-like): Test input feature matrix.
            y_test (array-like): Test target values.
            model_name (str): Name of the model for plotting purposes.

        Returns:
            w (array): Optimized coefficient vector.
            b (float): Optimized intercept.

        """
    Util.create_directory(Constants.plotting_path + model_name)
    n_features = X.shape[1]
    w = np.zeros(shape=n_features)
    b = 0
    total_samples = X.shape[0]

    accumulated_xs = []  # this will append each single xs on each iteration for plotting reasons
    accumulated_ys = []  # this will append each single ys on each iteration for plotting reasons

    for i in range(epochs):
        random_index = np.random.randint(total_samples)

        sample_x = X[random_index]
        sample_y = y[random_index]

        accumulated_xs.append(sample_x)
        accumulated_ys.append(sample_y)

        y_predicted = np.dot(w, sample_x.T) + b

        w_grad = 2 * sample_x * (y_predicted - sample_y)
        b_grad = 2 * (y_predicted - sample_y)

        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad

        Plotter.compute_acc_plot_per_iteration(X_train=X, y_train=y, w=w, b=b, iteration=i, X_test=X_test,
                                               y_test=y_test,
                                               accumulated_xs=accumulated_xs, accumulated_ys=accumulated_ys,
                                               model_name=model_name)

    return w, b




def stochastic_gradient_descent_plot_convergence2(X, y, epochs, learning_rate, X_test, y_test, model_name, modular):
    """
        Perform Stochastic Gradient Descent with convergence analysis and plot results.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the algorithm.
            X_test (array-like): Test input feature matrix.
            y_test (array-like): Test target values.
            model_name (str): Name of the model for plotting purposes.

        Returns:
            w (array): Optimized coefficient vector.
            b (float): Optimized intercept.

        """

    n_features = X.shape[1]
    w = np.zeros(shape=n_features)
    b = 0
    total_samples = X.shape[0]


    sgd_map = {}
    sgd_mse_map = {}
    accumulated_size = 0
    for i in range(epochs):
        random_index = np.random.randint(total_samples)

        sample_x = X[random_index]
        sample_y = y[random_index]

        accumulated_size +=1

        y_predicted = np.dot(w, sample_x.T) + b

        w_grad = 2 * sample_x * (y_predicted - sample_y)
        b_grad = 2 * (y_predicted - sample_y)

        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad

        if accumulated_size % modular == 0:
            y_predicted = Predictions.compute_predictions_(X_test, w, b)
            acc = Measures.r2_score_(y_test, y_predicted)
            sgd_map[accumulated_size] = "{:.5f}".format(acc)

            mse = np.mean((y_test - y_predicted)**2)
            sgd_mse_map[accumulated_size] = "{:.5f}".format(mse)


    return w, b, sgd_map, sgd_mse_map


def stochastic_gradient_descent_convergence2(X, y, epochs, learning_rate, model_name, seed, modular):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    sgd_list = []
    sgd_mse_list = []
    for fold_index, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, b, sgd_map, sgd_mse_map = stochastic_gradient_descent_plot_convergence2(X_train, y_train, epochs, learning_rate, X_test, y_test, model_name, modular)
        sgd_list.append(sgd_map)
        sgd_mse_list.append(sgd_mse_map)

    return sgd_list, sgd_mse_list
