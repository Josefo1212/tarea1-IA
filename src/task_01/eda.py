import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path.cwd()
CLEAN_CSV = BASE / "CSV" / "datos_limpios.csv"
OUT_DIR = BASE / "graficas"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN_CSV)
    for c in ['Popularity', 'Vote_Count', 'Vote_Average']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Popularity', 'Vote_Count', 'Vote_Average'])

    print("Shape:", df.shape)
    print(df[['Popularity','Vote_Count','Vote_Average']].describe().round(3))

    plt.figure(figsize=(6,4))
    sns.histplot(df['Vote_Average'], bins=30, kde=True)
    plt.title('Distribución de Vote_Average')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'hist_vote_average.png')

    plt.figure(figsize=(6,4))
    sns.scatterplot(x='Popularity', y='Vote_Average', data=df, alpha=0.2, s=10)
    plt.title('Popularity vs Vote_Average')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'scatter_popularity_vs_vote.png')

    plt.figure(figsize=(6,4))
    sns.scatterplot(x='Vote_Count', y='Vote_Average', data=df, alpha=0.2, s=10)
    plt.title('Vote_Count vs Vote_Average')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'scatter_votecount_vs_vote.png')

    plt.figure(figsize=(5,4))
    corr = df[['Popularity','Vote_Count','Vote_Average']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f')
    plt.title('Matriz de correlación')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'corr_heatmap.png')
    print("Gráficas guardadas en:", OUT_DIR)


if __name__ == "__main__":
    main()
