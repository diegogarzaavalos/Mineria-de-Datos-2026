import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# PREPARACION: serie anual de lanzamientos
# ---------------------------------------------------------------------------
df = pd.read_csv("dataset/game_data_clean.csv", encoding="utf-8")
df["release"] = pd.to_datetime(df["release"])

"""
Practica 8 — Pronostico
Pregunta: ¿cuantos juegos se lanzarian en años futuros si la tendencia
de lanzamientos por año de release siguiera una recta?
Cada fila del CSV es un juego, no un punto en el tiempo. La serie se
construye contando juegos por año de release.
2023 no entra al ajuste: el scrape fue a mitad de año, no al cierre.
"""

conteo = df.groupby(df["release"].dt.year).size().rename("n_juegos")
conteo.index.name = "anio"

anio_incompleto = 2023
serie = conteo[conteo.index < anio_incompleto]
observado_2023 = int(conteo.get(anio_incompleto, 0))

print("=== 1. SERIE ANUAL ===")
print(serie.to_string())
print(f"\n2023 observado (no se usa para entrenar): {observado_2023}")
print(f"Años de entrenamiento: {serie.index.min()} a {serie.index.max()}")
"""
2006-2013 tienen decenas o pocos cientos de lanzamientos; desde 2014
el conteo sube a miles. La recta resume esa tendencia larga, no cada salto.
"""

# ---------------------------------------------------------------------------
# 2. REGRESION n_juegos ~ anio
# ---------------------------------------------------------------------------
x = serie.index.to_numpy(dtype=float)
y = serie.to_numpy(dtype=float)
resultado = stats.linregress(x, y)
r2 = resultado.rvalue ** 2

print("\n=== 2. MODELO LINEAL ===")
print(f"n_juegos = {resultado.intercept:.1f} + {resultado.slope:.1f} * anio")
print(f"Pendiente (juegos por año adicional) = {resultado.slope:.1f}")
print(f"R^2 = {r2:.4f}")
print(f"r = {resultado.rvalue:.4f}")
"""
Pendiente ≈ 717 juegos adicionales por cada año de calendario, en el
ajuste 2006-2022.
R^2 ≈ 0.85: la recta recoge la subida larga (de decenas de juegos a
~11000). No describe el detalle: 2006-2013 estan casi planos y el
salto fuerte empieza en 2014. El intercepto es enorme y negativo
porque el año es un numero cercano a 2000; se lee la pendiente, no el
intercepto suelto.
"""

# ---------------------------------------------------------------------------
# 3. PRONOSTICO 2023-2025
# ---------------------------------------------------------------------------
anios_futuros = np.array([2023, 2024, 2025], dtype=float)
pred_futuro = resultado.intercept + resultado.slope * anios_futuros

print("\n=== 3. PRONOSTICO ===")
for anio, pred in zip(anios_futuros, pred_futuro):
    print(f"{int(anio)}: {pred:.0f} juegos")
print(f"2023 observado en el CSV: {observado_2023} (año incompleto, no comparable)")
"""
Pronostico de la recta: 2023 ≈ 9818, 2024 ≈ 10535, 2025 ≈ 11252 juegos.
2024 y 2025 no estan en el CSV.
2023 observado es 2544, muy por debajo de 9818, porque el scrape no
cubre el año completo. No se usa ese 2544 para decir que el modelo
fallo.
"""

# ---------------------------------------------------------------------------
# 4. GRAFICA
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(11, 5))

x_linea = np.linspace(x.min(), anios_futuros.max(), 200)
y_linea = resultado.intercept + resultado.slope * x_linea
ax.plot(x_linea, y_linea, color="#e76f51", linewidth=2, label="Recta ajustada (2006-2022)")
ax.scatter(x, y, color="#2a9d8f", s=40, zorder=3, label="Observado (entrenamiento)")
ax.scatter(
    anios_futuros,
    pred_futuro,
    color="#e9c46a",
    s=50,
    zorder=3,
    label="Pronóstico 2023-2025",
)
if observado_2023:
    ax.scatter(
        [anio_incompleto],
        [observado_2023],
        color="#264653",
        marker="x",
        s=60,
        zorder=4,
        label="2023 observado (incompleto)",
    )

ax.set_title("Pronóstico de lanzamientos por año")
ax.set_xlabel("Año de release")
ax.set_ylabel("Cantidad de juegos")
# La recta cruza bajo cero cerca de 2006; un conteo de juegos no es negativo.
ax.set_ylim(bottom=0)
ax.legend()
sns.despine()
plt.tight_layout()
plt.show()
"""
La recta sube porque el tramo 2014-2022 pesa mas que la meseta inicial.
Los puntos de 2006-2013 quedan por encima o por debajo de la recta:
el R^2 no es 1.
El 2023 observado (marca x) esta muy por debajo del pronostico de la recta
porque el año no esta completo en el scrape, no porque el modelo 'fallara'
contra un año cerrado.
"""