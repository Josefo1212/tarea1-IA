import pandas as pd
from pathlib import Path

BASE = Path.cwd()
CSV_PATH = BASE / "CSV" / "Top Movies dataset.csv"
OUT_DIR = BASE / "CSV"
OUT_PATH = OUT_DIR / "datos_limpios.csv"

def main():
    df = pd.read_csv(CSV_PATH)
    df['Popularity'] = pd.to_numeric(df['Popularity'], errors='coerce')
    df['Vote_Count'] = pd.to_numeric(df['Vote_Count'], errors='coerce')
    df['Vote_Average'] = pd.to_numeric(df['Vote_Average'], errors='coerce')
    df = df.dropna(subset=['Popularity', 'Vote_Count', 'Vote_Average'])
    df = df.reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Archivo limpio guardado en: {OUT_PATH}")
    print("Shape:", df.shape)

if __name__ == "__main__":
    main()
