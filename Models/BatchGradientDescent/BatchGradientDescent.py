import numpy as np
from sklearn.model_selection import KFold
from Utils import Measures, Predictions, Util, Plotter, Constants

"""
This python script represents the implementation of the Batch Gradient Descent
Please note that the batch gradient descent performance yield to the exact same
performance of Batch Regression (Pseudo-Inverse) which is used as the benchmark
for other models performance in our experiments.
"""

import numpy as np


def batch_gradient_descent(X, y_true, epochs, learning_rate):
    total_samples = X.shape[0]
    number_of_features = X.shape[1]
    w = np.zeros(number_of_features)
    b = 0

    cost_list = []
    epoch_list = []

    for i in range(epochs):
        y_predicted = np.dot(w, X.T) + b

        w_grad = -(2 / total_samples) * (X.T.dot(y_true - y_predicted))
        b_grad = -(2 / total_samples) * np.sum(y_true - y_predicted)

        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad

        cost = np.mean(np.square(y_true - y_predicted))  # MSE (Mean Squared Error)
        if i % 50 == 0:
            cost_list.append(cost)
            epoch_list.append(i)

    return w, b, cost_list, epoch_list

def mini_batch_gradient_descent_KFold(X, y, epochs, learning_rate, seed):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, b, epoch_list, cost_list = batch_gradient_descent(X_train, y_train, epochs, learning_rate)
        y_predicted = Predictions.compute_predictions_(X_test, w, b)
        acc_per_split_for_same_seed = Measures.r2_score_(y_test, y_predicted)
        scores.append(acc_per_split_for_same_seed)
    acc = np.array(scores).mean()
    return acc
