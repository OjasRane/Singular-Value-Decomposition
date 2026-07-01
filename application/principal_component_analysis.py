import numpy as np
from optimized_method import optimized_svd

class PCA:
    """
    Principal Component Analysis (PCA) via SVD.
    Reduces dimensionality by projecting data onto the directions of maximum variance.
    """
    def __init__(self, n_components):
        """
        :param n_components: number of components to keep
        """
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError(f"Expected n_components to be a positive integer, got {n_components}")
        self.n_components = n_components
        self._is_fitted = False
        self.n_samples = None
        self.n_features = None
        self.mean = None
        self.U = None
        self.S = None
        self.V = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X):
        """
        Fits the principal component analysis algorithm
        :param X: 2D numpy array of shape (n_samples, n_features)
        :return: self
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array got {X.ndim}D array")
        max_rank = min(X.shape)
        if self.n_components > max_rank:
            raise ValueError(f"n_components must be less than min(X.shape)={max_rank}")
        if X.shape[0] < 2:
            raise ValueError(f"Dataset must contain at least 2 samples")
        self.n_samples, self.n_features = X.shape
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        U, S, Vt = optimized_svd.optimized_svd(X)
        self.U = U[:, :self.n_components]
        self.V = Vt.T[:, :self.n_components]
        eigenvalues = np.square(S) / (self.n_samples - 1)
        self.S = S[:self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = eigenvalues[:self.n_components] / np.sum(eigenvalues)
        self._is_fitted = True
        return self

    def fit_transform(self, X):
        """
        Fits the principal component analysis algorithm and returns the transformed data
        :param X: 2D numpy array of shape (n_samples, n_features)
        :return: Transformed 2D numpy array of shape (n_samples, n_components)
        """
        self.fit(X)
        return self.U * self.S

    def transform(self, X):
        """
        Transforms input data X according to the fitted model.
        :param X: 2D numpy array of shape (samples, n_features)
        :return: Transformed 2D numpy array of shape (samples, n_components)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before transform")
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array got {X.ndim}D array")
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features but got {X.shape[1]} features")
        X = X - self.mean
        return X @ self.V

    def inverse_transform(self, X):
        """
        Reconstructs transform data X
        :param X: 2D numpy array of shape (samples, n_components)
        :return: 2D numpy array of shape (samples, n_features)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before inverse_transform")
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array got {X.ndim}D array")
        if X.shape[1] != self.n_components:
            raise ValueError(f"Expected {self.n_components} components but got {X.shape[1]} components")
        return X @ self.V.T + self.mean