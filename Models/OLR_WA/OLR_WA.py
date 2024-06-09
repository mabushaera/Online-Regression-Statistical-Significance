from Models.BatchGradientDescent import BatchGradientDescent
"""
OLR-WA OnLine Regression with Weighted Average.

This script implements OLR-WA Online regression with weighted average.
It leverages tools from the scikit-learn library and custom utility functions for various tasks such as performance
evaluation, plotting, and data manipulation.

Functions:
- `olr_wa_regression`: Perform olr_wa to optimize linear regression coefficients.
- `olr_wa_regression_KFold`: Perform K-Fold cross-validation with olr_wa for linear regression.
- `olr_wa_regression_adversarial`: Perform olr_wa for linear regression and evaluate on adversarial test data.
- `olr_wa_regression_convergence`: Perform olr_wa for linear regression with convergence analysis.
- `olr_wa_plot_convergence`: Perform olr_wa with convergence analysis and plot results.

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
Date: Aug, 2023
"""
import numpy as np
from Utils import Measures, Util, Predictions, Plotter, Constants
from Models.BatchRegression import BatchRegression
from HyperPlanesUtil import PlanesIntersection, PlaneDefinition
from sklearn.model_selection import KFold


def olr_wa_regression(X, y, w_base, w_inc, base_model_size, increment_size, no_of_base_model_points=None):
    """
        Perform Online Linear Regression with Weighted Averaging (OLR-WA).

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            w_base (float): user defined w_base weigh, higher value will favor the base model which represents the data
            seen so far.
            w_inc (float): user defined w_inc weight, higher value will favor new data.
            base_model_size (int): percent of the total like (1, or 10) which represents 1% or 10% samples for base model.
            increment_size (int): Number of samples representing the incremental mini-batch.

        Returns:
            w (array): Optimized coefficient vector using Weighted Averaging.
        """
    n_samples, n_features = X.shape

    cost_list = np.array([])
    epoch_list = np.array([])

    # Step 1: base-regression = pseudo-inverse(base-X,base-y)
    # Calculate the linear regression for the base model, the base model
    # is a percent of all the data, usually 10% of all the data.
    # the outcome of step 1 in Alg is w_base.
    if no_of_base_model_points is None:
        no_of_base_model_points = Util.calculate_no_of_base_model_points(n_samples, base_model_size)

    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]
    r_w_base = BatchRegression.linear_regression(base_model_training_X, base_model_training_y)
    base_model_predicted_y = Predictions._compute_predictions_(base_model_training_X, r_w_base)
    base_coeff = np.array(np.append(np.append(r_w_base[1:], -1), r_w_base[0]))

    cost = np.mean(np.square(base_model_training_y - base_model_predicted_y))
    cost_list = np.append(cost_list, cost)
    epoch_list = np.append(epoch_list, no_of_base_model_points)

    # Step 2: for t ← 1 to T do
    # In this step we look over the rest of the data incrementally with a determined
    # increment size. In this experiment we use increment_size = max(3, (n+1) * 5) where n is the number of features.

    for i in range(no_of_base_model_points, n_samples - no_of_base_model_points, increment_size):
        # Step 3: inc-regression = pseudo-inverse(inc-X,in-y)
        # Calculate the linear regression for each increment model
        # (for the no of points on each increment increment_size)
        Xj = X[i:i + increment_size]
        yj = y[i:i + increment_size]
        r_w_inc = BatchRegression.linear_regression(Xj, yj)
        inc_predicted_y = Predictions._compute_predictions_(Xj, r_w_inc)
        inc_coeff = np.array(np.append(np.append(r_w_inc[1:], -1), r_w_inc[0]))

        # Step 4: v-avg1 = (w-base · v-base + w-inc · v-inc)/(w-base + w-inc)
        #         v-avg2 = (-1 · w-base · v-base + w-inc · v-inc)/(w-base + w-inc)

        n1 = base_coeff[:-1]
        n2 = inc_coeff[:-1]
        d1 = base_coeff[-1]
        d2 = inc_coeff[-1]

        # in case the base and the incremental models are coincident
        if PlanesIntersection.isCoincident(n1, n2, d1, d2): continue

        n1norm = n1 / np.sqrt((n1 * n1).sum())  # normalization
        n2norm = n2 / np.sqrt((n2 * n2).sum())  # normalization
        avg1 = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)
        # avg2 = (np.dot(-w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)

        # avg1 = ((base_coeff * w_base)  + (inc_coeff*w_inc)) / (w_base + w_inc)
        # r_w_base = ((r_w_base * w_base) + (r_w_inc * w_inc)) / (w_base + w_inc)
        # base_coeff = avg1
        # # avg2 = (np.dot(-w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)
        #
        # # Step 5: intersection-point = get-intersection-point(base-regression, inc-regression)
        # # We will find an intersection point between the two models, the base and the incremental.
        # # if no intersection point, then the two hyperplanes are parallel, then, the intersection point
        # # will be a weighted middle point.
        intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1=n1, n2=n2, d1=d1, d2=d2,
                                                                               w_base=w_base, w_inc=w_inc)
        #
        # # Step 6: space-coeff-1 = define-new-space(v-avg1, intersection-point)
        # #         space-coeff-2 = define-new-space(v-avg2, intersection-point)
        # # In this step we define two new spaces as a result from the average vector 1 and the intersection point
        # # and from the average vector 2, and the intersection point
        #
        avg1_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg1, intersection_point)
        # avg2_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg2, intersection_point)
        #
        # # Step 7 : err-v1= MSE(space-coeff-1, Xcombined, ycombined)
        # #          err-v2= MSE(space-coeff-2, Xcombined, ycombined)
        # # MSE method has an inside call to the sample_and_combine method which samples from base_coeff and combine with
        # # current data Xj, and yj
        # r_sq1 = Measures.MSE(Xj, yj, avg1_plane, base_coeff)
        # r_sq2 = Measures.MSE(Xj, yj, avg2_plane, base_coeff)
        # if (r_sq1 < r_sq2):
        #     base_coeff = avg1_plane
        # else:
        #     base_coeff = avg2_plane
        base_coeff = avg1_plane
        inc_predicted_y_test = Predictions._compute_predictions__(Xj, base_coeff)
        cost = np.mean(np.square(yj - inc_predicted_y_test))
        cost_list = np.append(cost_list, cost)
        epoch_list = np.append(epoch_list, i+no_of_base_model_points)
    base_coeff = np.array(np.append(np.append(r_w_base[1:], -1), r_w_base[0]))

    return base_coeff, epoch_list, cost_list


def olr_wa_regression_(X, y, w_base, w_inc, base_model_size, increment_size, no_of_base_model_points=None):
    """
        Perform Online Linear Regression with Weighted Averaging (OLR-WA).

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            w_base (float): user defined w_base weigh, higher value will favor the base model which represents the data
            seen so far.
            w_inc (float): user defined w_inc weight, higher value will favor new data.
            base_model_size (int): percent of the total like (1, or 10) which represents 1% or 10% samples for base model.
            increment_size (int): Number of samples representing the incremental mini-batch.

        Returns:
            w (array): Optimized coefficient vector using Weighted Averaging.
        """
    n_samples, n_features = X.shape

    cost_list = np.array([])
    epoch_list = np.array([])

    # Step 1: base-regression = pseudo-inverse(base-X,base-y)
    # Calculate the linear regression for the base model, the base model
    # is a percent of all the data, usually 10% of all the data.
    # the outcome of step 1 in Alg is w_base.
    if no_of_base_model_points is None:
        no_of_base_model_points = Util.calculate_no_of_base_model_points(n_samples, base_model_size)
    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]
    r_w_base = BatchRegression.linear_regression_(base_model_training_X, base_model_training_y)
    base_model_predicted_y = Predictions._compute_predictions_(base_model_training_X, r_w_base)
    base_coeff = np.array(np.append(np.append(r_w_base[1:], -1), r_w_base[0]))

    cost = np.mean(np.square(base_model_training_y - base_model_predicted_y))
    cost_list = np.append(cost_list, cost)
    epoch_list = np.append(epoch_list, no_of_base_model_points)

    # Step 2: for t ← 1 to T do
    # In this step we look over the rest of the data incrementally with a determined
    # increment size. In this experiment we use increment_size = max(3, (n+1) * 5) where n is the number of features.

    for i in range(no_of_base_model_points, n_samples - no_of_base_model_points, increment_size):
        # Step 3: inc-regression = pseudo-inverse(inc-X,in-y)
        # Calculate the linear regression for each increment model
        # (for the no of points on each increment increment_size)
        Xj = X[i:i + increment_size]
        yj = y[i:i + increment_size]
        r_w_inc = BatchRegression.linear_regression_(Xj, yj)
        inc_predicted_y = Predictions._compute_predictions_(Xj, r_w_inc)
        inc_coeff = np.array(np.append(np.append(r_w_inc[1:], -1), r_w_inc[0]))

        # Step 4: v-avg1 = (w-base · v-base + w-inc · v-inc)/(w-base + w-inc)
        #         v-avg2 = (-1 · w-base · v-base + w-inc · v-inc)/(w-base + w-inc)

        n1 = base_coeff[:-1]
        n2 = inc_coeff[:-1]
        d1 = base_coeff[-1]
        d2 = inc_coeff[-1]

        # in case the base and the incremental models are coincident
        if PlanesIntersection.isCoincident(n1, n2, d1, d2): continue

        n1norm = n1 / np.sqrt((n1 * n1).sum())  # normalization
        n2norm = n2 / np.sqrt((n2 * n2).sum())  # normalization
        avg1 = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)
        avg2 = (np.dot(-w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)

        # Step 5: intersection-point = get-intersection-point(base-regression, inc-regression)
        # We will find an intersection point between the two models, the base and the incremental.
        # if no intersection point, then the two hyperplanes are parallel, then, the intersection point
        # will be a weighted middle point.
        intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1=n1, n2=n2, d1=d1, d2=d2,
                                                                               w_base=w_base, w_inc=w_inc)

        # Step 6: space-coeff-1 = define-new-space(v-avg1, intersection-point)
        #         space-coeff-2 = define-new-space(v-avg2, intersection-point)
        # In this step we define two new spaces as a result from the average vector 1 and the intersection point
        # and from the average vector 2, and the intersection point

        avg1_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg1, intersection_point)
        avg2_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg2, intersection_point)

        # Step 7 : err-v1= MSE(space-coeff-1, Xcombined, ycombined)
        #          err-v2= MSE(space-coeff-2, Xcombined, ycombined)
        # MSE method has an inside call to the sample_and_combine method which samples from base_coeff and combine with
        # current data Xj, and yj
        r_sq1 = Measures.MSE(Xj, yj, avg1_plane, base_coeff)
        r_sq2 = Measures.MSE(Xj, yj, avg2_plane, base_coeff)
        if (r_sq1 < r_sq2):
            base_coeff = avg1_plane
        else:
            base_coeff = avg2_plane

        inc_predicted_y_test = Predictions._compute_predictions__(Xj, base_coeff)
        cost = np.mean(np.square(yj - inc_predicted_y_test))
        cost_list = np.append(cost_list, cost)
        epoch_list = np.append(epoch_list, i)

    return base_coeff, epoch_list, cost_list


def olr_wa_regression_KFold(X, y, w_base, w_inc, base_model_size, increment_size, seed, no_of_base_model_points):
    """
        Perform K-Fold cross-validation with Online Linear Regression and Weighted Averaging.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            w_base (float): user defined w_base weigh, higher value will favor the base model which represents the data
            seen so far.
            w_inc (float): user defined w_inc weight, higher value will favor new data.
            base_model_size (int): Percent of total samples for base model (1% or 10%).
            increment_size (int): Number of samples representing the incremental mini-batch.
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

        w, epoch_list, cost_list = olr_wa_regression(X_train, y_train, w_base, w_inc, base_model_size,
                                                     increment_size, no_of_base_model_points)

        predicted_y_test = Predictions._compute_predictions__(X_test, w)
        acc = Measures.r2_score_(y_test, predicted_y_test)
        scores.append(acc)

        avg_cost = np.array(cost_list).mean()
        costs.append(avg_cost)



    return np.array(scores).mean(), np.array(costs).mean()



def olr_wa_regression_KFold_(X, y, w_base, w_inc, base_model_size, increment_size, seed, no_of_base_model_points):
    """
        Perform K-Fold cross-validation with Online Linear Regression and Weighted Averaging.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            w_base (float): user defined w_base weigh, higher value will favor the base model which represents the data
            seen so far.
            w_inc (float): user defined w_inc weight, higher value will favor new data.
            base_model_size (int): Percent of total samples for base model (1% or 10%).
            increment_size (int): Number of samples representing the incremental mini-batch.
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

        w, epoch_list, cost_list = olr_wa_regression_(X_train, y_train, w_base, w_inc, base_model_size,
                                                     increment_size, no_of_base_model_points)

        predicted_y_test = Predictions._compute_predictions__(X_test, w)
        acc = Measures.r2_score_(y_test, predicted_y_test)
        scores.append(acc)
        avg_cost = np.array(cost_list).mean()
        costs.append(avg_cost)

    return np.array(scores).mean(), np.array(costs).mean()

def olr_wa_regression_adversarial(X_train, y_train, w_base, w_inc, base_model_size, increment_size, X_test, y_test):
    """
        Evaluate Online Linear Regression with Weighted Averaging on adversarial test data.

        Args:
            X_train (array-like): Input feature matrix for training.
            y_train (array-like): Target values for training.
            w_base (float): user defined w_base weigh, higher value will favor the base model which represents the data
            seen so far.
            w_inc (float): user defined w_inc weight, higher value will favor new data.
            base_model_size (int): Percent of total samples for base model (1% or 10%).
            increment_size (int): Number of samples representing the incremental mini-batch.
            X_test (array-like): Adversarial test input feature matrix.
            y_test (array-like): Adversarial test target values.

        Returns:
            acc (float): Accuracy (R-squared) on adversarial test data.
        """
    w, epoch_list, cost_list = olr_wa_regression(X_train, y_train, w_base, w_inc, base_model_size,
                                                 increment_size)
    predicted_y_test = Predictions._compute_predictions__(X_test, w)
    acc = Measures.r2_score_(y_test, predicted_y_test)

    return acc


def olr_wa_regression_convergence(X, y, w_base, w_inc, base_model_size, increment_size):
    """
        Perform convergence analysis for Online Linear Regression with Weighted Averaging.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            w_base (float): user defined w_base weigh, higher value will favor the base model which represents the data
            seen so far.
            w_inc (float): user defined w_inc weight, higher value will favor new data.
            base_model_size (int): Percent of total samples for base model (1% or 10%).
            increment_size (int): Number of samples representing the incremental mini-batch.

        Returns:
            acc (float): Mean accuracy (R-squared) across multiple runs.
            epochs_accu (array): Array of epochs for convergence analysis.
            cost_accu (array): Array of cost values for convergence analysis.
        """
    n_splits = 5
    kf = KFold(n_splits)
    scores = []
    epoch_list_per_seed = np.array([])
    cost_list_per_seed = np.array([])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        w, epoch_list, cost_list = olr_wa_regression(X_train, y_train,
                                                     w_base,
                                                     w_inc,
                                                     base_model_size,
                                                     increment_size)

        epoch_list_per_seed = Util.sum_lists_element_wise(epoch_list_per_seed, epoch_list)
        cost_list_per_seed = Util.sum_lists_element_wise(cost_list_per_seed, cost_list)
        predicted_y_test = Predictions._compute_predictions__(X_test, w)
        acc = Measures.r2_score_(y_test, predicted_y_test)
        scores.append(acc)

    acc = np.array(scores).mean()
    epochs_accu = epoch_list_per_seed / n_splits
    cost_accu = cost_list_per_seed / n_splits

    return acc, epochs_accu, cost_accu


def olr_wa_plot_convergence(X_train, y_train, X_test, y_test, w_base, w_inc, base_model_size, increment_size,
                            model_name):
    """
        Plot the convergence and computes acc (R-Squared) per iteration of Online Linear Regression with Weighted Averaging.

        Args:
            X_train (array-like): Input feature matrix for training.
            y_train (array-like): Target values for training.
            X_test (array-like): Test input feature matrix.
            y_test (array-like): Test target values.
            w_base (float): user defined w_base weigh, higher value will favor the base model which represents the data
            seen so far.
            w_inc (float): user defined w_inc weight, higher value will favor new data.
            base_model_size (int): Percent of total samples for base model (1% or 10%).
            increment_size (int): Number of samples representing the incremental mini-batch.
            model_name (str): Name of the model for saving plots.

        Returns:
            w (array): Optimized coefficient vector using Weighted Averaging.
        """
    Util.create_directory(Constants.plotting_path + model_name)
    n_samples = X_train.shape[0] + X_test.shape[0]

    # Step 1: base-regression = pseudo-inverse(base-X,base-y)
    # Calculate the linear regression for the base model, the base model
    # is a percent of all the data, usually 10% of all the data.
    # the outcome of step 1 in Alg is w_base.
    no_of_base_model_points = Util.calculate_no_of_base_model_points(n_samples, base_model_size)
    base_model_training_X = X_train[:no_of_base_model_points]
    base_model_training_y = y_train[:no_of_base_model_points]
    r_w_base = BatchRegression.linear_regression(base_model_training_X, base_model_training_y)
    base_model_predicted_y = Predictions._compute_predictions_(base_model_training_X, r_w_base)
    acc_base = Measures.r2_score_(base_model_training_y, base_model_predicted_y)
    base_coeff = np.array(np.append(np.append(r_w_base[1:], -1), r_w_base[0]))

    # showing convergence speed
    accumulated_xs = np.array([])
    accumulated_ys = np.array([])
    accumulated_xs = np.append(accumulated_xs, base_model_training_X)
    accumulated_ys = np.append(accumulated_ys, base_model_training_y)
    Plotter.compute_acc_plot_per_iteration(X_train=X_train, y_train=y_train, w=base_coeff, b=None,
                                           iteration=no_of_base_model_points, X_test=X_test, y_test=y_test,
                                           accumulated_xs=accumulated_xs, accumulated_ys=accumulated_ys,
                                           model_name=model_name)

    # Step 2: for t ← 1 to T do
    # In this step we look over the rest of the data incrementally with a determined
    for i in range(no_of_base_model_points, X_train.shape[0] - no_of_base_model_points, increment_size):
        # Step 3: inc-regression = pseudo-inverse(inc-X,in-y)
        # Calculate the linear regression for each increment
        Xj = X_train[i:i + increment_size]
        yj = y_train[i:i + increment_size]

        accumulated_xs = np.append(accumulated_xs, np.array(Xj).flatten())
        accumulated_ys = np.append(accumulated_ys, np.array(yj).flatten())

        r_w_inc = BatchRegression.linear_regression(Xj, yj)
        inc_predicted_y = Predictions._compute_predictions_(Xj, r_w_inc)
        inc_coeff = np.array(np.append(np.append(r_w_inc[1:], -1), r_w_inc[0]))

        # Step 4: v-avg1 = (w-base · v-base + w-inc · v-inc)/(w-base + w-inc)
        #         v-avg2 = (-1 · w-base · v-base + w-inc · v-inc)/(w-base + w-inc)
        n1 = base_coeff[:-1]
        n2 = inc_coeff[:-1]
        d1 = base_coeff[-1]
        d2 = inc_coeff[-1]

        # in case the base and the incremental models are coincident
        if PlanesIntersection.isCoincident(n1, n2, d1, d2): continue

        n1norm = n1 / np.sqrt((n1 * n1).sum())
        n2norm = n2 / np.sqrt((n2 * n2).sum())
        avg1 = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)
        avg2 = (np.dot(-w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)

        # Step 5: intersection-point = get-intersection-point(base-regression, inc-regression)
        # We will find an intersection point between the two models, the base and the incremental.
        # if no intersection point, then the two hyperplanes are parallel, then, the intersection point
        # will be a weighted middle point.
        intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1, n2, d1, d2, w_base, w_inc)

        # Step 6: space-coeff-1 = define-new-space(v-avg1, intersection-point)
        #         space-coeff-2 = define-new-space(v-avg2, intersection-point)
        # In this step we define two new spaces as a result from the average vector 1 and the intersection point
        # and from the average vector 2, and the intersection point
        avg1_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg1, intersection_point)
        avg2_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg2, intersection_point)

        # Step 7 : err-v1= MSE(space-coeff-1, Xcombined, ycombined)
        #          err-v2= MSE(space-coeff-2, Xcombined, ycombined)
        # MSE method has an inside call to the sample_and_combine method which samples from base_coeff and combine with
        # current data Xj, and yj
        r_sq1 = Measures.MSE(Xj, yj, avg1_plane, base_coeff)
        r_sq2 = Measures.MSE(Xj, yj, avg2_plane, base_coeff)
        if r_sq1 < r_sq2:
            base_coeff = avg1_plane
        else:
            base_coeff = avg2_plane

        Plotter.compute_acc_plot_per_iteration(X_train=X_train, y_train=y_train, w=base_coeff, b=None,
                                               iteration=len(accumulated_xs), X_test=X_test, y_test=y_test,
                                               accumulated_xs=accumulated_xs, accumulated_ys=accumulated_ys,
                                               model_name=model_name)

    return base_coeff



def olr_wa_plot_convergence2(X_train, y_train, X_test, y_test, w_base, w_inc, base_model_size, increment_size,
                            model_name):
    n_samples = X_train.shape[0] + X_test.shape[0]
    olr_wa_map = {}
    olr_wa_mse_map = {}

    # Step 1: base-regression = pseudo-inverse(base-X,base-y)
    # Calculate the linear regression for the base model, the base model
    # is a percent of all the data, usually 10% of all the data.
    # the outcome of step 1 in Alg is w_base.


    # no_of_base_model_points = Util.calculate_no_of_base_model_points(n_samples, base_model_size)
    no_of_base_model_points = base_model_size
    base_model_training_X = X_train[:no_of_base_model_points]
    base_model_training_y = y_train[:no_of_base_model_points]
    r_w_base = BatchRegression.linear_regression_(base_model_training_X, base_model_training_y)
    base_model_predicted_y = Predictions._compute_predictions_(base_model_training_X, r_w_base)
    acc_base = Measures.r2_score_(base_model_training_y, base_model_predicted_y)
    base_coeff = np.array(np.append(np.append(r_w_base[1:], -1), r_w_base[0]))

    # showing convergence speed
    accumulated_xs = np.array([])
    accumulated_ys = np.array([])
    accumulated_xs = np.append(accumulated_xs, base_model_training_X)
    accumulated_ys = np.append(accumulated_ys, base_model_training_y)
    # Plotter.compute_acc_plot_per_iteration(X_train=X_train, y_train=y_train, w=base_coeff, b=None,
    #                                        iteration=no_of_base_model_points, X_test=X_test, y_test=y_test,
    #                                        accumulated_xs=accumulated_xs, accumulated_ys=accumulated_ys,
    #                                        model_name=model_name)

    y_predicted = Predictions._compute_predictions__(X_test, base_coeff)
    y_predicted = np.array(y_predicted).flatten()
    acc = Measures.r2_score_(y_test, y_predicted)
    olr_wa_map[no_of_base_model_points] = "{:.5f}".format(acc)

    mse = np.mean((y_test - y_predicted) ** 2)
    olr_wa_mse_map[no_of_base_model_points] = "{:.5f}".format(mse)

    accumulated_data_size = no_of_base_model_points
    # Step 2: for t ← 1 to T do
    # In this step we look over the rest of the data incrementally with a determined
    for i in range(no_of_base_model_points, X_train.shape[0] - no_of_base_model_points, increment_size):
        # Step 3: inc-regression = pseudo-inverse(inc-X,in-y)
        # Calculate the linear regression for each increment
        Xj = X_train[i:i + increment_size]
        yj = y_train[i:i + increment_size]

        accumulated_data_size +=Xj.shape[0]

        r_w_inc = BatchRegression.linear_regression_(Xj, yj)
        inc_predicted_y = Predictions._compute_predictions_(Xj, r_w_inc)
        inc_coeff = np.array(np.append(np.append(r_w_inc[1:], -1), r_w_inc[0]))

        # Step 4: v-avg1 = (w-base · v-base + w-inc · v-inc)/(w-base + w-inc)
        #         v-avg2 = (-1 · w-base · v-base + w-inc · v-inc)/(w-base + w-inc)
        n1 = base_coeff[:-1]
        n2 = inc_coeff[:-1]
        d1 = base_coeff[-1]
        d2 = inc_coeff[-1]

        # in case the base and the incremental models are coincident
        if PlanesIntersection.isCoincident(n1, n2, d1, d2): continue

        n1norm = n1 / np.sqrt((n1 * n1).sum())
        n2norm = n2 / np.sqrt((n2 * n2).sum())
        avg1 = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)
        # avg2 = (np.dot(-w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)

        # Step 5: intersection-point = get-intersection-point(base-regression, inc-regression)
        # We will find an intersection point between the two models, the base and the incremental.
        # if no intersection point, then the two hyperplanes are parallel, then, the intersection point
        # will be a weighted middle point.
        intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1, n2, d1, d2, w_base, w_inc)

        # Step 6: space-coeff-1 = define-new-space(v-avg1, intersection-point)
        #         space-coeff-2 = define-new-space(v-avg2, intersection-point)
        # In this step we define two new spaces as a result from the average vector 1 and the intersection point
        # and from the average vector 2, and the intersection point
        avg1_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg1, intersection_point)
        # avg2_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg2, intersection_point)

        # Step 7 : err-v1= MSE(space-coeff-1, Xcombined, ycombined)
        #          err-v2= MSE(space-coeff-2, Xcombined, ycombined)
        # MSE method has an inside call to the sample_and_combine method which samples from base_coeff and combine with
        # current data Xj, and yj
        # r_sq1 = Measures.MSE(Xj, yj, avg1_plane, base_coeff)
        # r_sq2 = Measures.MSE(Xj, yj, avg2_plane, base_coeff)
        # if r_sq1 < r_sq2:
        #     base_coeff = avg1_plane
        # else:
        #     base_coeff = avg2_plane

        base_coeff = avg1_plane

        y_predicted = Predictions._compute_predictions__(X_test, base_coeff)
        y_predicted = np.array(y_predicted).flatten()
        acc = Measures.r2_score_(y_test, y_predicted)
        olr_wa_map[accumulated_data_size] = "{:.5f}".format(acc)

        mse = np.mean((y_test - y_predicted) ** 2)
        olr_wa_mse_map[accumulated_data_size] = "{:.5f}".format(mse)

    return base_coeff, olr_wa_map, olr_wa_mse_map



def olr_wa_convergence2(X, y, w_base, w_inc, base_model_size, increment_size,model_name, seed):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    olr_wa_list = []
    olr_wa_mse_list = []
    for fold_index, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, olr_wa_map, olr_wa_mse_map = olr_wa_plot_convergence2(X_train, y_train, X_test, y_test, w_base, w_inc, base_model_size, increment_size,
                            model_name)
        olr_wa_list.append(olr_wa_map)
        olr_wa_mse_list.append(olr_wa_mse_map)

    return olr_wa_list, olr_wa_mse_list




