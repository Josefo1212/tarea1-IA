## Uso

Para ejecutar todo (limpiar dataset + comparación):

```bash
cd /home/josefo1212/JfPrograms/AI/task-01
uv run task-01
```

Para ejecutar el módulo directamente:

```bash
cd /home/josefo1212/JfPrograms/AI/task-01
uv run --no-project python -m task_01
```

Para generar solo el archivo limpio:

```bash
cd /home/josefo1212/JfPrograms/AI/task-01
uv run --no-project python src/task_01/cleanFile.py
```

Para ejecutar el EDA (genera gráficas en graficas/):

```bash
cd /home/josefo1212/JfPrograms/AI/task-01
uv run --no-project python -m task_01.eda
```

## Archivos
- src/task_01/cleanFile.py: lee CSV/Top Movies dataset.csv, limpia y genera CSV/datos_limpios.csv
- src/task_01/MyLinearRegression.py: implementación con ecuación normal (fit/predict/mse/r2_score), atributos b_ e intercept_
- src/task_01/compare.py: compara con sklearn.LinearRegression usando CSV/datos_limpios.csv
- src/task_01/eda.py: análisis exploratorio usando CSV/datos_limpios.csv (guarda gráficas en graficas/)
- src/task_01/clean_and_compare.py: orquesta cleanFile + compare
- src/task_01/__main__.py: permite `python -m task_01`
- src/task_01/__init__.py: exporta main para `uv run task-01`
