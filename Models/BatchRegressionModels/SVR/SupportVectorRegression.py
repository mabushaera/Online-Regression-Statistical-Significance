import numpy as np
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
# from sklearn.preprocessing import StandardScaler


def support_vector_regression(X_train, y_train, C=100, epsilon=0.1, kernel='linear'):

    # Create SVR model
    svr = SVR(C=C, epsilon=epsilon, kernel=kernel)

    # Fit model
    svr.fit(X_train, y_train)

    return np.array(svr.coef_).flatten(), svr.intercept_[0]