import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# PREPARACION
# ---------------------------------------------------------------------------
df = pd.read_csv("dataset/game_data_clean.csv", encoding="utf-8")

# derivadas de P2 (por si el CSV no las trae)
if "total_reviews" not in df.columns:
    df["total_reviews"] = df["positive_reviews"] + df["negative_reviews"]

"""
Practica 5 — Modelos lineales y correlacion
Pregunta: ¿cuanto del pico de jugadores (all_time_peak) se explica
con el volumen de reseñas (total_reviews)?
Modelo: log10(all_time_peak) = a + b * log10(total_reviews)
Escala log: la lineal completa aplasta la nube en el origen (P3).
"""

# log exige valores > 0
df_modelo = df[(df["total_reviews"] > 0) & (df["all_time_peak"] > 0)].copy()
x = np.log10(df_modelo["total_reviews"].to_numpy(dtype=float))
y = np.log10(df_modelo["all_time_peak"].to_numpy(dtype=float))

print("=== 1. DATOS DEL MODELO ===")
print("Filas usadas:", len(df_modelo))
print("X: log10(total_reviews)")
print("Y: log10(all_time_peak)")
"""
Se modela en log10 para que la recta describa la tendencia de la
dispersion de P3, no el aplastamiento de unos pocos hits en escala lineal.
"""

# ---------------------------------------------------------------------------
# 2. AJUSTE LINEAL Y R^2
# ---------------------------------------------------------------------------
print("\n=== 2. REGRESION LINEAL (log-log) ===")
resultado = stats.linregress(x, y)
# rvalue es la correlacion de Pearson; R^2 = r^2
r2 = resultado.rvalue ** 2
print(f"Pendiente (b) = {resultado.slope:.4f}")
print(f"Intercepto (a) = {resultado.intercept:.4f}")
print(f"Correlacion de Pearson (r) = {resultado.rvalue:.4f}")
print(f"R^2 = {r2:.4f}")
print(f"p-value de la pendiente = {resultado.pvalue:.4e}")
"""
Ecuacion: log10(peak) = a + b * log10(reviews).
R^2 = fraccion de la variacion de Y (en log) explicada por X (en log).
r cerca de 1 y positivo = asociacion positiva; el p-value solo dice si
la pendiente es distinguible de cero (con n grande casi siempre lo es).
"""

# prediccion y residuos para las graficas
y_pred = resultado.intercept + resultado.slope * x
residuos = y - y_pred

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# 3. DISPERSION + RECTA AJUSTADA
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(x, y, alpha=0.15, s=8, c="#264653", edgecolors="none")
x_linea = np.linspace(x.min(), x.max(), 200)
y_linea = resultado.intercept + resultado.slope * x_linea
ax.plot(
    x_linea,
    y_linea,
    color="#e76f51",
    linewidth=2,
    label=f"Ajuste: y = {resultado.intercept:.2f} + {resultado.slope:.2f}x\nR² = {r2:.3f}",
)
ax.set_xlabel("log10(total_reviews)")
ax.set_ylabel("log10(all_time_peak)")
ax.set_title("Modelo lineal: pico vs reseñas (escala log)")
ax.legend(loc="upper left")
sns.despine()
plt.tight_layout()
plt.show()
"""
La nube sube con la recta: a mas reseñas (en log), mayor pico (en log).
Pendiente b ≈ 0.87: en esta escala el pico crece un poco menos que
proporcionalmente a las reseñas (b = 1 seria crecimiento proporcional).
r ≈ 0.86 (asociacion positiva fuerte).
R^2 ≈ 0.73: las reseñas explican cerca de tres cuartas partes de la
variacion del pico en log; el resto queda sin explicar por este modelo.
El p-value de la pendiente es practicamente 0 por el n grande: indica
que la pendiente no es cero, no que el ajuste sea perfecto.
"""

# ---------------------------------------------------------------------------
# 4. RESIDUOS
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(y_pred, residuos, alpha=0.15, s=8, c="#264653", edgecolors="none")
ax.axhline(0, color="#e76f51", linestyle="--", linewidth=1.5)
ax.set_xlabel("log10(all_time_peak) predicho")
ax.set_ylabel("Residuo (observado - predicho)")
ax.set_title("Residuos del modelo lineal (log-log)")
sns.despine()
plt.tight_layout()
plt.show()
"""
Residuo = log10(peak) observado menos el que predice la recta.
La linea en 0 marca el ajuste exacto para ese juego.
Los residuos se concentran alrededor de 0, coherente con un R^2 ≈ 0.73,
pero hay dispersion vertical: juegos con las mismas reseñas pueden tener
picos distintos. No se afirma causalidad (las reseñas no "causan" el pico);
ambas crecen con la popularidad del juego (misma lectura que la dispersion de P3).
"""

print("\n=== FIN Practica 5 ===")
