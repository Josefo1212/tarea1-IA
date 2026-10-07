import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

try:
    from .MyLinearRegression import MyLinearRegression
except Exception:
    from MyLinearRegression import MyLinearRegression

BASE = Path.cwd()
CLEAN_CSV = BASE / "CSV" / "datos_limpios.csv"


def main():
    df = pd.read_csv(CLEAN_CSV)
    for c in ['Popularity', 'Vote_Count', 'Vote_Average']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Popularity', 'Vote_Count', 'Vote_Average'])

    X = df[['Popularity', 'Vote_Count']].values
    y = df['Vote_Average'].values

    model = MyLinearRegression()
    model.fit(X, y)
    pred = model.predict(X)

    print("=== MyLinearRegression ===")
    print("intercept_:", model.intercept_)
    print("b_ (coef):", model.b_)
    print("MSE:", model.mse(y, pred))
    print("R2:", model.r2_score(y, pred))

    sk = LinearRegression(fit_intercept=True)
    sk.fit(X, y)
    pred_sk = sk.predict(X)
    print("\n=== sklearn LinearRegression ===")
    print("intercept_:", sk.intercept_)
    print("coef_:", sk.coef_)
    print("MSE:", mean_squared_error(y, pred_sk))
    print("R2:", r2_score(y, pred_sk))

    print("\n=== Diferencias ===")
    print("diff_intercept:", abs(model.intercept_ - sk.intercept_))
    print("diff_coef_max:", np.max(np.abs(model.b_ - sk.coef_)))


if __name__ == "__main__":
    main()
