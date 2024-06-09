from sklearn import datasets


def create_dataset(n_samples, n_features, noise, shuffle, random_state):
    """
        Create a synthetic regression dataset using scikit-learn's datasets module.

        This function generates a synthetic regression dataset with specified parameters using
        scikit-learn's `make_regression` function. It creates a feature matrix (X) and target vector (y)
        suitable for regression tasks.

        Parameters:
            n_samples (int): Number of samples in the dataset.
            n_features (int): Number of features in each sample.
            noise (float): Standard deviation of the noise added to the target values.
            shuffle (bool): Whether to shuffle the samples randomly.
            random_state (int or None): Seed for the random number generator.

        Returns:
            X (numpy.ndarray): Feature matrix (X) of shape (n_samples, n_features).
            y (numpy.ndarray): Target vector (y) of shape (n_samples,).
        """
    X, y = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise,
                                    shuffle=shuffle, random_state=random_state)
    return X, y



