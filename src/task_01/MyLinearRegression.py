import numpy as np


class MyLinearRegression:
    """Regresión lineal multivariable con la ecuación normal: theta = (X^T X)^-1 X^T y"""

    def __init__(self):
        self.b_ = None          
        self.intercept_ = None  

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xb = np.c_[np.ones(X.shape[0]), X]         
        theta = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y  
        self.intercept_ = theta[0]
        self.b_ = theta[1:]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.b_ + self.intercept_

    def mse(self, y_true, y_pred):
        y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return 1 - ss_res / ss_tot