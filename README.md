# Minería de Datos

Repositorio de prácticas y proyecto integrador (PIA) de la materia **Minería de Datos**.

## Dataset

**Nombre:** Steam Releases (juegos de Steam)

**Fuente:** [Steam Releases — Kaggle](https://www.kaggle.com/datasets/whigmalwhim/steam-releases?select=game_data_all.csv)

**Descripción:** Conjunto de datos con información de juegos publicados en Steam: fechas de lanzamiento, reseñas, géneros, publisher, developer, tecnologías detectadas y picos de jugadores concurrentes.

### Cumplimiento de requisitos del curso

| Requisito | Detalle |
|-----------|---------|
| Variables | 12 (dataset limpio) |
| Numéricas | `ID`, `positive_reviews`, `negative_reviews`, `all_time_peak` |
| Categóricas / texto | `game`, `primary_genre`, `store_genres`, `publisher`, `developer`, `detected_technologies` |
| Fechas | `release`, `all_time_peak_date` |
| Filas | 59,775 (tras limpieza) |
| Archivo único | `dataset/game_data_clean.csv` |

### Archivos del dataset

| Archivo | Descripción |
|---------|-------------|
| `dataset/game_data_all.csv` | Dataset original descargado de Kaggle |
| `dataset/game_data_clean.csv` | Dataset limpio generado en la Práctica 1 |

## Estructura del repositorio

```
Mineria-de-Datos-2026/
├── dataset/
│   ├── game_data_all.csv
│   └── game_data_clean.csv
├── Practica 1/
│   └── limpieza_dataset.py
└── README.md
```

## Práctica 1 — Limpieza de Datos

**Script:** `Practica 1/limpieza_dataset.py`

**Acciones realizadas:**
- Renombrado de `Unnamed: 0` a `ID` (identificador único de cada registro)
- Eliminación de columnas redundantes o no relevantes para el análisis
- Detección y eliminación de valores nulos
- Conversión de `release` y `all_time_peak_date` a tipo fecha
- Validación de valores negativos en variables numéricas
- Revisión de fechas de pico anteriores al lanzamiento (early access)
- Estandarización de valores en `primary_genre`
- Exportación del dataset limpio

**Ejecución:**

```bash
python "Practica 1/limpieza_dataset.py"
```

**Dependencias:** `pandas`

## Prácticas

| Práctica | Tema | Estado |
|----------|------|--------|
| 1 | Limpieza de Datos | ✅ Completada |
| 2 | Estadística Descriptiva | ⏳ Pendiente |
| 3 | Visualización de Datos | ⏳ Pendiente |
| 4 | Pruebas Estadísticas | ⏳ Pendiente |
| 5 | Modelos Lineales y Correlación | ⏳ Pendiente |
| 6 | Clasificación KNN | ⏳ Pendiente |
| 7 | Clustering K-Means | ⏳ Pendiente |
| 8 | Forecasting | ⏳ Pendiente |
| 9 | Análisis de Texto | ⏳ Pendiente |
| PIA | Reporte de Inteligencia de Negocios | ⏳ Pendiente |
