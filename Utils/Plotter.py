import numpy as np
from Utils import Predictions, Measures, Constants
import matplotlib.pyplot as plt


def plot(X_train, y_train, iteration, X_test, y_test, accumulated_xs, accumulated_ys, model_name, y_predicted):
    """
        Create and save a plot showing the training data, accumulated points, and model predictions.

        Args:
            X_train (array-like): Training input features.
            y_train (array-like): Training actual target values.
            iteration (int): Iteration number.
            X_test (array-like): Test input features.
            y_test (array-like): Test actual target values.
            accumulated_xs (list): Accumulated x-coordinates.
            accumulated_ys (list): Accumulated y-coordinates.
            model_name (str): Name of the model.
            y_predicted (array-like): Predicted target values.

        Returns:
            None
        """
    n_features = X_train.shape[1]
    if n_features == 1:
        plt.figure()
        plt.title(model_name + " - iteration " + str(iteration))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.scatter(X_train, y_train, color='blue')
        plt.scatter(accumulated_xs, accumulated_ys, color='green')
        plt.plot(X_test, y_predicted, color='red')
        comment = "itr: " + str(iteration)
        x_position = X_test[-1]  # x-coordinate for the comment (end of the line)
        y_position = y_predicted[-1]  # y-coordinate for the comment (end of the line)
        plt.annotate(comment, (x_position, y_position), xytext=(10, -15),
                     textcoords='offset points', color='red')
        plot_name = model_name + str(iteration) + '.png'
        path = Constants.plotting_path + model_name + '/'
        plt.savefig(path + plot_name)
        # plt.show() # will show you the plot of each iteration (if you prefer this option)
        plt.close()


def compute_acc_plot_per_iteration(X_train, y_train, w, b, iteration, X_test, y_test, accumulated_xs,
                                   accumulated_ys, model_name):
    """
        Compute accuracy, create a plot, and save it at specified iterations.

        Args:
            X_train (array-like): Training input features.
            y_train (array-like): Training actual target values.
            w (array-like): Model's weight parameters.
            b (float): Model's bias parameter.
            iteration (int): Iteration number.
            X_test (array-like): Test input features.
            y_test (array-like): Test actual target values.
            accumulated_xs (list): Accumulated x-coordinates.
            accumulated_ys (list): Accumulated y-coordinates.
            model_name (str): Name of the model.

        Returns:
            None
        """
    accumulated_xs = np.array(accumulated_xs)
    accumulated_ys = np.array(accumulated_ys)

    if model_name == Constants.MODEL_NAME_LMS:
        accumulated_xs = accumulated_xs[:, 1]

    accumulated_xs = accumulated_xs.flatten()
    accumulated_ys = accumulated_ys.flatten()

    y_predicted = []
    if iteration % 10 == 0:

        if model_name == Constants.MODEL_NAME_OLW_WA:
            y_predicted = Predictions._compute_predictions__(X_test, w)
            y_predicted = np.array(y_predicted).flatten()
        if any(model_name == name for name in
               [Constants.MODEL_NAME_SGD, Constants.MODEL_NAME_MBGD, Constants.MODEL_NAME_ORR,
                Constants.MODEL_NAME_OLR]):
            y_predicted = Predictions.compute_predictions_(X_test, w, b)
        if model_name == Constants.MODEL_NAME_LMS:
            y_predicted = Predictions._compute_predictions_(X_test, w)
        if any(model_name == name for name in [Constants.MODEL_NAME_RLS, Constants.MODEL_NAME_PA]):
            y_predicted = Predictions.compute_predictions(X_test, w)

        acc = Measures.r2_score_(y_test, y_predicted)
        print('iteration', iteration, 'acc: ', "{:.5f}".format(acc))


        if iteration <= 200:
            plot(X_train, y_train, iteration, X_test, y_test, accumulated_xs, accumulated_ys, model_name, y_predicted)




def compute_acc(X_train, y_train, w, b, iteration, X_test, y_test, model_name):
    """
        Compute accuracy, create a plot, and save it at specified iterations.

        Args:
            X_train (array-like): Training input features.
            y_train (array-like): Training actual target values.
            w (array-like): Model's weight parameters.
            b (float): Model's bias parameter.
            iteration (int): Iteration number.
            X_test (array-like): Test input features.
            y_test (array-like): Test actual target values.
            accumulated_xs (list): Accumulated x-coordinates.
            accumulated_ys (list): Accumulated y-coordinates.
            model_name (str): Name of the model.

        Returns:
            None
        """



    y_predicted = []
    if iteration % 10 == 0:

        if model_name == Constants.MODEL_NAME_OLW_WA:
            y_predicted = Predictions._compute_predictions__(X_test, w)
            y_predicted = np.array(y_predicted).flatten()
        if any(model_name == name for name in
               [Constants.MODEL_NAME_SGD, Constants.MODEL_NAME_MBGD, Constants.MODEL_NAME_ORR,
                Constants.MODEL_NAME_OLR]):
            y_predicted = Predictions.compute_predictions_(X_test, w, b)
        if model_name == Constants.MODEL_NAME_LMS:
            y_predicted = Predictions._compute_predictions_(X_test, w)
        if any(model_name == name for name in [Constants.MODEL_NAME_RLS, Constants.MODEL_NAME_PA]):
            y_predicted = Predictions.compute_predictions(X_test, w)

        acc = Measures.r2_score_(y_test, y_predicted)
        print('iteration', iteration, 'acc: ', "{:.5f}".format(acc))
