# Minería de Datos

Repositorio de prácticas y proyecto integrador (PIA) de la materia **Minería de Datos**.

## Dataset

**Nombre:** Steam Releases (juegos de Steam)

**Fuente:** [Steam Releases — Kaggle](https://www.kaggle.com/datasets/whigmalwhim/steam-releases?select=game_data_all.csv)

**Descripción:** Conjunto de datos con información de juegos publicados en Steam: fechas de lanzamiento, reseñas, géneros, publisher, developer, tecnologías detectadas y picos de jugadores concurrentes.

### Cumplimiento de requisitos del curso

| Requisito | Detalle |
|-----------|---------|
| Variables | 14 (tras Práctica 2) |
| Numéricas | `ID`, `positive_reviews`, `negative_reviews`, `all_time_peak`, `total_reviews`, `review_positive_percentage` |
| Categóricas / texto | `game`, `primary_genre`, `store_genres`, `publisher`, `developer`, `detected_technologies` |
| Fechas | `release`, `all_time_peak_date` |
| Filas | 59,775 |
| Archivo único | `dataset/game_data_clean.csv` |

### Archivos del dataset

| Archivo | Descripción |
|---------|-------------|
| `dataset/game_data_all.csv` | Dataset original descargado de Kaggle |
| `dataset/game_data_clean.csv` | Dataset limpio (P1) enriquecido en P2 con `total_reviews` y `review_positive_percentage` |

`release` se mantiene como **fecha completa**. El año de análisis se obtiene en código con `release.dt.year` (no se guarda una columna `release_year`).

## Estructura del repositorio

```
Mineria-de-Datos-2026/
├── dataset/
│   ├── game_data_all.csv
│   └── game_data_clean.csv
├── Practica 1/
│   └── limpieza_dataset.py
├── Practica 2/
│   ├── estadistica_descriptiva.py
│   └── ERD Steam Releases.pdf
├── changelog.txt
└── README.md
```

## Línea de análisis

Reseñas (positivas, negativas, total y proporción positiva) según el **año de lanzamiento** y el **género principal**, contrastando tops de publishers/developers por **popularidad** (suma de reseñas) vs **productividad** (cantidad de juegos).

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
- Exportación del dataset limpio (12 columnas)

**Ejecución:**

```bash
python "Practica 1/limpieza_dataset.py"
```

**Dependencias:** `pandas`

## Práctica 2 — Estadística Descriptiva

**Script:** `Practica 2/estadistica_descriptiva.py`  
**Diagrama:** `Practica 2/ERD Steam Releases.pdf`  
**Dataset:** lee y actualiza `dataset/game_data_clean.csv`

**Acciones realizadas:**
- Columnas derivadas persistidas en el CSV: `total_reviews`, `review_positive_percentage`
- Año de lanzamiento vía `release.dt.year` (sin columna `release_year`)
- Estadística descriptiva de variables numéricas (incluye medianas)
- Frecuencias y porcentajes de `primary_genre`
- Conteo de lanzamientos por año de `release`
- Métricas agrupadas por año × género principal
- Tops de publisher y developer por suma de reseñas y por cantidad de juegos
- Identificación de entidades y relaciones (ERD)

**Ejecución:**

```bash
python "Practica 2/estadistica_descriptiva.py"
```

**Dependencias:** `pandas`

## Prácticas

| Práctica | Tema | Estado |
|----------|------|--------|
| 1 | Limpieza de Datos | ✅ Completada |
| 2 | Estadística Descriptiva | ✅ Completada |
| 3 | Visualización de Datos | ⏳ Pendiente |
| 4 | Pruebas Estadísticas | ⏳ Pendiente |
| 5 | Modelos Lineales y Correlación | ⏳ Pendiente |
| 6 | Clasificación KNN | ⏳ Pendiente |
| 7 | Clustering K-Means | ⏳ Pendiente |
| 8 | Forecasting | ⏳ Pendiente |
| 9 | Análisis de Texto | ⏳ Pendiente |
| PIA | Reporte de Inteligencia de Negocios | ⏳ Pendiente |
