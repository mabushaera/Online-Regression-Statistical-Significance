import numpy as np
import sklearn.metrics

from Utils import Measures, Util, Predictions, Plotter, Constants
from Models.BatchGradientDescent import BatchGradientDescent
from Models.BatchRegressionModels.SVR import SupportVectorRegression
from HyperPlanesUtil import PlanesIntersection, PlaneDefinition
from sklearn.model_selection import KFold

def olr_wa_regression_KFold_BGD(X, y, w_base, w_inc, base_model_size, increment_size, seed):
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
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        w, epoch_list, cost_list = olr_wa_regression_BGD(X_train, y_train, w_base, w_inc, base_model_size,
                                                     increment_size)

        predicted_y_test = Predictions._compute_predictions__(X_test, w)
        acc = Measures.r2_score_(y_test, predicted_y_test)
        scores.append(acc)

    return np.array(scores).mean()


def olr_wa_regression_BGD(X, y, w_base, w_inc, base_model_size, increment_size):
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
    no_of_base_model_points = Util.calculate_no_of_base_model_points(n_samples, base_model_size)
    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]
    #r_w_base = BatchRegression.linear_regression(base_model_training_X, base_model_training_y)
    r_w_base, r_b_base,cost_list, epoch_list = BatchGradientDescent.batch_gradient_descent(base_model_training_X, base_model_training_y, 500, .01)
    # print('r_w_base:', r_w_base, 'r_b_base:', r_b_base)
    base_model_predicted_y = Predictions.compute_predictions_(base_model_training_X, r_w_base, r_b_base)
    # base_coeff = np.array(np.append(np.append(r_w_base[1:], -1), r_w_base[0]))
    base_coeff = np.array(np.append(np.append(r_w_base, -1), r_b_base))


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
        r_w_inc, r_b_inc, cost_list, epoch_list = BatchGradientDescent.batch_gradient_descent(Xj, yj, 500, 0.01)
        # print('r_w_inc:', r_w_inc, ' r_b_inc', r_b_inc )
        inc_predicted_y = Predictions.compute_predictions_(Xj, r_w_inc, r_b_inc)
        inc_coeff = np.array(np.append(np.append(r_w_inc, -1), r_b_inc))

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
        epoch_list = np.append(epoch_list, i+no_of_base_model_points)

    return base_coeff, epoch_list, cost_list



def olr_wa_regression_KFold_SVR(X, y, w_base, w_inc, base_model_size, increment_size, seed):
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
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        w, epoch_list, cost_list = olr_wa_regression_SVR(X_train, y_train, w_base, w_inc, base_model_size,
                                                     increment_size)

        predicted_y_test = Predictions._compute_predictions__(X_test, w)
        acc = Measures.r2_score_(y_test, predicted_y_test)
        scores.append(acc)

    return np.array(scores).mean()


def olr_wa_regression_SVR(X, y, w_base, w_inc, base_model_size, increment_size):
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
    no_of_base_model_points = Util.calculate_no_of_base_model_points(n_samples, base_model_size)
    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]
    #r_w_base = BatchRegression.linear_regression(base_model_training_X, base_model_training_y)
    # r_w_base, r_b_base,cost_list, epoch_list = BatchGradientDescent.batch_gradient_descent(base_model_training_X, base_model_training_y, 500, .01)

    r_w_base, r_b_base = SupportVectorRegression.support_vector_regression(base_model_training_X, base_model_training_y)
    base_model_predicted_y = Predictions.compute_predictions_XX(base_model_training_X, r_w_base, r_b_base)
    base_coeff = np.array(np.append(np.append(r_w_base, -1), r_b_base))


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
        # r_w_inc, r_b_inc, cost_list, epoch_list = BatchGradientDescent.batch_gradient_descent(Xj, yj, 500, 0.01)
        r_w_inc, r_b_inc = SupportVectorRegression.support_vector_regression(
            Xj, yj)
        # print('r_w_inc:::', r_w_inc, 'r_b_inc:::', r_b_inc)
        inc_predicted_y = Predictions.compute_predictions_XX(Xj, r_w_inc, r_b_inc)
        inc_coeff = np.array(np.append(np.append(r_w_inc, -1), r_b_inc))

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
        epoch_list = np.append(epoch_list, i+no_of_base_model_points)

    return base_coeff, epoch_list, cost_list

##################################################


def olr_wa_regression_KFold_MAX_LIKELIHOOD(X, y, w_base, w_inc, base_model_size, increment_size, seed):
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
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        w, epoch_list, cost_list = olr_wa_regression_max_likelihood(X_train, y_train, w_base, w_inc, base_model_size,
                                                     increment_size)

        predicted_y_test = Predictions._compute_predictions__(X_test, w)
        acc = Measures.r2_score_(y_test, predicted_y_test)
        scores.append(acc)

    return np.array(scores).mean()


def olr_wa_regression_max_likelihood(X, y, w_base, w_inc, base_model_size, increment_size):
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
    no_of_base_model_points = Util.calculate_no_of_base_model_points(n_samples, base_model_size)
    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]
    # r_w_base = BatchRegression.linear_regression(base_model_training_X, base_model_training_y)
    from Models.BatchRegressionModels.MaximumLikelihood import LinearRegressionMaxLikelihood

    r_w_base, r_b_base = LinearRegressionMaxLikelihood.linear_regression_mle(base_model_training_X, base_model_training_y)
    base_model_predicted_y = np.dot(base_model_training_X, r_w_base) + r_b_base
    base_coeff = np.array(np.append(np.append(r_w_base, -1), r_b_base))

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
        # r_w_inc = BatchRegression.linear_regression(Xj, yj)
        r_w_inc, r_b_base = LinearRegressionMaxLikelihood.linear_regression_mle(Xj, yj)
        # inc_predicted_y = Predictions._compute_predictions_(Xj, r_w_inc)
        # inc_predicted_y = Xj.dot(r_w_inc)
        inc_predicted_y = np.dot(Xj, r_w_inc) + r_b_base
        inc_coeff = np.array(np.append(np.append(r_w_inc, -1), r_b_base))

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
        epoch_list = np.append(epoch_list, i+no_of_base_model_points)

    return base_coeff, epoch_list, cost_list