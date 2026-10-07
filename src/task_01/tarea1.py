
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mylinearregression import MyLinearRegression

df = pd.read_csv('Top Movies dataset.csv')
print("filas y columnas del dataset: ", df.shape)
df.info()
print(df.describe().round(2))

print("Valores faltantes por columna:")
print(df.isna().sum())
print("Filas duplicadas:", df.duplicated().sum())

df["Vote_Count"] = pd.to_numeric(df["Vote_Count"], errors="coerce")
df["Vote_Average"] = pd.to_numeric(df["Vote_Average"], errors="coerce")
print("Filas con Vote_Average = 0:", (df["Vote_Average"] == 0).sum())

df = df.drop_duplicates().dropna()
df = df[df["Vote_Average"] > 0]
print("Dimensiones después de limpiar:", df.shape)
print("Faltantes restantes:", df.isna().sum().sum())
print("Duplicados restantes:", df.duplicated().sum())

for col in ["Popularity", "Vote_Count", "Vote_Average"]:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    out = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
    print(col, "-> outliers:", len(out))

df["log_Popularity"] = np.log1p(df["Popularity"])
df["log_Vote_Count"] = np.log1p(df["Vote_Count"])
df.to_csv("datos_limpios.csv", index=False)

cols = ["Popularity", "Vote_Count", "Vote_Average"]
cols_all = cols + ["log_Popularity", "log_Vote_Count"]

df[cols_all].hist(bins=30, figsize=(12, 8))
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, col in zip(axes, cols):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title("Boxplot de " + col)
plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df[cols_all].corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Matriz de correlación"); plt.show()

for x in ["Popularity", "Vote_Count", "log_Popularity", "log_Vote_Count"]:
    sns.regplot(data=df, x=x, y="Vote_Average",
                scatter_kws={"alpha": 0.3, "s": 10}, line_kws={"color": "red"})
    plt.title(x + " vs Vote_Average"); plt.show()

print(df[cols_all].corr()["Vote_Average"].sort_values(ascending=False))

X = df[["log_Popularity", "log_Vote_Count"]]
y = df["Vote_Average"]

mio = MyLinearRegression().fit(X, y)
sk = LinearRegression().fit(X, y)
p_mio, p_sk = mio.predict(X), sk.predict(X)
print("\n=== Entrenando con todo el dataset ===")
print("b_ (mía):      ", mio.b_, "| intercept_:", mio.intercept_)
print("coef_ (sklearn):", sk.coef_, "| intercept_:", sk.intercept_)
print("MSE  mía:", mio.mse(y, p_mio), "| sklearn:", mean_squared_error(y, p_sk))
print("R2   mía:", mio.r2_score(y, p_mio), "| sklearn:", r2_score(y, p_sk))
print("Dif. máx. coeficientes:", np.abs(mio.b_ - sk.coef_).max())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

mio = MyLinearRegression().fit(X_train, y_train)
sk = LinearRegression().fit(X_train, y_train)

print("\n=== Partición 80/20 ===")
for nombre, m in [("Mía", mio), ("sklearn", sk)]:
    ptr, pte = m.predict(X_train), m.predict(X_test)
    print(f"{nombre:8s} Train: MSE={mean_squared_error(y_train, ptr):.4f} "
          f"R2={r2_score(y_train, ptr):.4f} | "
          f"Test: MSE={mean_squared_error(y_test, pte):.4f} "
          f"R2={r2_score(y_test, pte):.4f}")
print("Varianza de y (referencia):", y.var())
