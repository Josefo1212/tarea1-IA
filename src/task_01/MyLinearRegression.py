import numpy as np


class MyLinearRegression:
    """Regresión lineal multivariable mediante la ecuación normal:

        theta = (X^T X)^{-1} X^T y

    Atributos (siguiendo la convención de sklearn):
        b_         : array (n_features,) con los parámetros de las características.
        intercept_ : float, el intercepto (theta_0).
    """

    def __init__(self):
        self.b_ = None
        self.intercept_ = None

    @staticmethod
    def _to_2d(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def fit(self, X, y):
        """Ajusta el modelo a los datos de entrenamiento."""
        X = self._to_2d(X)
        y = np.asarray(y, dtype=float).ravel()

        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y deben tener el mismo número de filas.")

        # Agregar columna de unos para el intercepto: X_b = [1, X]
        X_b = np.c_[np.ones(X.shape[0]), X]

        # Ecuación normal: theta = (X^T X)^{-1} X^T y
        XtX = X_b.T @ X_b
        try:
            theta = np.linalg.inv(XtX) @ X_b.T @ y
        except np.linalg.LinAlgError:
            # X^T X singular (características colineales): usar pseudoinversa
            theta = np.linalg.pinv(XtX) @ X_b.T @ y

        self.intercept_ = theta[0]
        self.b_ = theta[1:]
        return self

    def predict(self, X):
        """Realiza predicciones sobre nuevos datos."""
        if self.b_ is None:
            raise RuntimeError("El modelo no ha sido ajustado. Llama a fit() primero.")
        X = self._to_2d(X)
        return X @ self.b_ + self.intercept_

    def mse(self, y_true, y_pred):
        """Error cuadrático medio."""
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        """Coeficiente de determinación R^2 = 1 - SS_res / SS_tot."""
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1 - ss_res / ss_tot