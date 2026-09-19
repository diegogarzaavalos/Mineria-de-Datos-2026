import pandas as pd
import numpy as np
from itertools import combinations
from scipy import stats

# ---------------------------------------------------------------------------
# PREPARACION
# ---------------------------------------------------------------------------
df = pd.read_csv("dataset/game_data_clean.csv", encoding="utf-8")

# derivadas de P2 (por si el CSV no las trae)
if "total_reviews" not in df.columns:
    df["total_reviews"] = df["positive_reviews"] + df["negative_reviews"]
if "review_positive_percentage" not in df.columns:
    df["review_positive_percentage"] = (
        df["positive_reviews"] / df["total_reviews"].replace(0, pd.NA)
    )

ALPHA = 0.05
"""
Practica 4 — Pruebas estadisticas
Pregunta: ¿el % positivo de reseñas difiere entre los top 8 primary_genre?
H0: las distribuciones (o medias, segun el test) son iguales entre generos.
H1: al menos un genero difiere.
alpha = 0.05
"""

# top 8 generos (misma logica que P3: segmentos con mayor peso en el catalogo)
freq_genero = df["primary_genre"].value_counts()
top8_generos = freq_genero.head(8).index.tolist()
df_test = df[df["primary_genre"].isin(top8_generos)].copy()
df_test = df_test.dropna(subset=["review_positive_percentage"])

print("=== 1. DATOS PARA LA PRUEBA ===")
print("Generos (top 8):", top8_generos)
print("Filas usadas:", len(df_test))
print("\nTamaño y mediana de % positivo por genero:")
resumen = (
    df_test.groupby("primary_genre", observed=True)["review_positive_percentage"]
    .agg(n="count", mediana="median", media="mean", std="std")
    .reindex(top8_generos)
)
print(resumen.round(4))
"""
Se usan solo top 8 para:
* coherencia con el pastel/boxplot de P3
* evitar generos con n muy bajo (poca potencia / comparaciones inestables)
La metrica es review_positive_percentage (acotada 0-1), no el volumen de reseñas.
"""

# listas de valores por grupo (orden = top8)
grupos = [
    df_test.loc[df_test["primary_genre"] == g, "review_positive_percentage"].to_numpy()
    for g in top8_generos
]

# ---------------------------------------------------------------------------
# 2. SUPUESTOS — ¿ANOVA o Kruskal-Wallis?
# ---------------------------------------------------------------------------
print("\n=== 2. REVISION DE SUPUESTOS ===")

# Levene: homogeneidad de varianzas (robusto a no normalidad)
stat_levene, p_levene = stats.levene(*grupos, center="median")
print(f"Levene (center=median): stat={stat_levene:.4f}, p={p_levene:.4e}")
print(
    "  -> Varianzas ",
    "NO homogeneas (p < alpha)" if p_levene < ALPHA else "homogeneas (p >= alpha)",
)

# Normalidad: con n miles, Shapiro casi siempre rechaza; se prueba una muestra
print("\nShapiro-Wilk (muestra n=500 por genero; orientativo con n grande):")
for g, vals in zip(top8_generos, grupos):
    muestra = np.random.default_rng(42).choice(vals, size=min(500, len(vals)), replace=False)
    stat_sw, p_sw = stats.shapiro(muestra)
    print(f"  {g}: W={stat_sw:.4f}, p={p_sw:.4e}")
"""
Decision del test:
* En P2/P3 el % positivo tiene outliers bajos y cajas asimétricas por genero.
* Levene y Shapiro (muestras) suelen indicar que ANOVA clasico no es ideal
  (normalidad + homocedasticidad dudosas).
* Por eso se elige Kruskal-Wallis (no parametrico): compara distribuciones/
  rangos entre k grupos etiquetados sin exigir normalidad.
* ANOVA + t se reserva cuando los supuestos se cumplen; aqui no es el caso.
"""

# ---------------------------------------------------------------------------
# 3. KRUSKAL-WALLIS (prueba global)
# ---------------------------------------------------------------------------
print("\n=== 3. KRUSKAL-WALLIS ===")
stat_kw, p_kw = stats.kruskal(*grupos)
print(f"H = {stat_kw:.4f}")
print(f"p = {p_kw:.4e}")
print(f"alpha = {ALPHA}")
if p_kw < ALPHA:
    print("Decision: se RECHAZA H0 (hay diferencias entre al menos dos generos).")
else:
    print("Decision: NO se rechaza H0 (no hay evidencia de diferencias).")
"""
Kruskal-Wallis responde si ALGUN genero difiere en la distribucion del
% positivo. Un p significativo no dice CUALES pares difieren: eso va en
las pruebas post-hoc por pares.
"""

# ---------------------------------------------------------------------------
# 4. POST-HOC: Mann-Whitney por pares + Bonferroni
# ---------------------------------------------------------------------------
print("\n=== 4. COMPARACIONES POR PARES (Mann-Whitney + Bonferroni) ===")
pares = list(combinations(range(len(top8_generos)), 2))
n_pares = len(pares)
alpha_bonf = ALPHA / n_pares
print(f"Numero de pares: {n_pares}")
print(f"alpha Bonferroni = {ALPHA}/{n_pares} = {alpha_bonf:.6f}")
"""
Mann-Whitney: analogo no parametrico de la prueba t para 2 grupos.
Bonferroni: divide alpha entre el numero de comparaciones para controlar
el error tipo I familiar (muchas pruebas a la vez).
"""

filas_posthoc = []
for i, j in pares:
    g1, g2 = top8_generos[i], top8_generos[j]
    # alternative two-sided; nan_policy omitido (ya sin NaN)
    u_stat, p_raw = stats.mannwhitneyu(grupos[i], grupos[j], alternative="two-sided")
    significativo = p_raw < alpha_bonf
    filas_posthoc.append({
        "genero_A": g1,
        "genero_B": g2,
        "U": u_stat,
        "p_raw": p_raw,
        "significativo_bonf": significativo,
        "mediana_A": np.median(grupos[i]),
        "mediana_B": np.median(grupos[j]),
    })

posthoc = pd.DataFrame(filas_posthoc)
n_sig = posthoc["significativo_bonf"].sum()
print(f"Pares significativos (Bonferroni): {n_sig} / {n_pares}")

# mostrar solo pares significativos, ordenados por p
sig = posthoc[posthoc["significativo_bonf"]].sort_values("p_raw")
print("\nPares con diferencia significativa:")
if len(sig) == 0:
    print("  (ninguno tras correccion Bonferroni)")
else:
    print(
        sig[
            ["genero_A", "genero_B", "mediana_A", "mediana_B", "p_raw"]
        ].to_string(index=False)
    )

print("\n--- Resumen de TODOS los pares (primeros 15 por p_raw) ---")
print(
    posthoc.sort_values("p_raw")[
        ["genero_A", "genero_B", "mediana_A", "mediana_B", "p_raw", "significativo_bonf"]
    ]
    .head(15)
    .to_string(index=False)
)
"""
Interpretacion metodologica (no de negocio):
* Kruskal significativo + varios pares Bonferroni significativos implica
  que las diferencias de % positivo entre generos no son solo ruido muestral.
* Con n muy grande, diferencias pequenas en mediana pueden salir significativas:
  conviene mirar tambien la magnitud (medianas A vs B), no solo el p-value.
* Coherente con el boxplot de P3: medianas cercanas (~0.75-0.82) pero
  con IQR/outliers distintos; el test detecta esas diferencias de distribucion.
"""

