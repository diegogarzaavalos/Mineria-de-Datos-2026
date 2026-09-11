import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

#cargar dataset (P1 limpio; derivadas se calculan aqui por si el CSV no las trae)
df = pd.read_csv("dataset/game_data_clean.csv", encoding="utf-8")
df["release"] = pd.to_datetime(df["release"])

#mismas derivadas que en Practica 2 (necesarias para dispersion, boxplot y barras)
if "total_reviews" not in df.columns:
    df["total_reviews"] = df["positive_reviews"] + df["negative_reviews"]
if "review_positive_percentage" not in df.columns:
    df["review_positive_percentage"] = (
        df["positive_reviews"] / df["total_reviews"].replace(0, pd.NA)
    )

sns.set_theme(style="whitegrid")
"""
Practica 3 — Visualizacion de datos
8 figuras, 5 tipos: pastel, boxplot, lineas, dispersion, barras.
Ciclo en rankings de barras (automatizacion).
Orden: pastel -> boxplot (mismos top 8) -> lineas -> dispersion -> barras.
"""

# ---------------------------------------------------------------------------
# 1) PASTEL — top 8 primary_genre (+ Otros)
# ---------------------------------------------------------------------------
freq_genero = df["primary_genre"].value_counts()
top8_generos = freq_genero.head(8).index
top8 = freq_genero.head(8)
otros = freq_genero.iloc[8:].sum()
pastel = pd.concat([top8, pd.Series({"Otros": otros})])

plt.figure(figsize=(9, 9))
plt.pie(pastel.values, labels=pastel.index, autopct="%1.1f%%", startangle=90)
plt.title("Top 8 genero principal + Otros")
plt.tight_layout()
plt.show()
"""
Pastel top 8 para legibilidad; 'Otros' agrupa el resto del catalogo.
El numero al lado del genero es el ID del genero en la API de Steam.

Analisis del pastel (composicion del catalogo):
* Indie (~26.9%) y Action (~21.8%) dominan: juntos casi la mitad (~48.7%).
* Adventure y Casual forman un segundo nivel (~12% cada uno).
* Strategy, Simulation y RPG rondan ~6% cada uno; Free to Play ~2.9%.
* Los top 8 concentran ~94.6% de los juegos; 'Otros' solo ~5.4%.

Por eso el boxplot siguiente usa exactamente estos top 8: comparar el
% positivo entre los segmentos que realmente importan en tamanio de catalogo,
sin diluir el grafico con generos muy poco frecuentes.
"""

# ---------------------------------------------------------------------------
# 2) BOXPLOT — % positivo por primary_genre (mismos top 8 del pastel)
# ---------------------------------------------------------------------------
df_box = df[df["primary_genre"].isin(top8_generos)].copy()

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(
    data=df_box,
    x="primary_genre",
    y="review_positive_percentage",
    order=top8_generos,
    color="#90be6d",
    ax=ax,
)
ax.set_title("% positivo de reseñas por género principal (top 8)")
ax.set_xlabel("primary_genre")
ax.set_ylabel("review_positive_percentage")
ax.tick_params(axis="x", rotation=30)
sns.despine()
plt.tight_layout()
plt.show()
"""
Analisis del boxplot (% positivo por genero, top 8):

Medianas (juego tipico por genero):
* Mas altas: Indie y Adventure (~0.82)
* Intermedias: Action y RPG (~0.79), Strategy (~0.78).
* Mas bajas del grupo: Free to Play (~0.76) y Simulation (~0.75).
* En todos los casos la mediana supera ~0.75: la recepcion tipica es positiva.

Dispersion (altura de la caja = IQR, 50% central):
* Simulation tiene la caja mas baja en Q1 (~0.56) y mediana menor: mayor
  variabilidad hacia reseñas menos favorables.
* Free to Play tambien muestra IQR amplio y mediana algo menor.
* Indie, Adventure y RPG se ven un poco mas compactos en la zona alta.

Extremos:
* Bigote superior llega a 1.0 en todos los generos (hay juegos 100% positivos).
* Bigotes inferiores bajan ~0.1-0.25 segun el genero.
* Outliers densos cerca de 0.0-0.2 en todos los segmentos: existen juegos
  con % positivo muy bajo aunque el tipico este alto.

Puente con el pastel:
* Indie/Action dominan el volumen del catalogo, pero el % positivo tipico
  no cambia de forma drastica entre generos (todas las medianas en un rango
  estrecho ~0.75-0.82). La gran diferencia entre generos es de cantidad de
  juegos, no de una brecha enorme en recepcion mediana.
"""

# ---------------------------------------------------------------------------
# 3) LINEAS — lanzamientos por año de release
# ---------------------------------------------------------------------------
juegos_por_anio = df.groupby(df["release"].dt.year).size()

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(
    juegos_por_anio.index,
    juegos_por_anio.values,
    marker="o",
    color="#2a9d8f",
    linewidth=2,
)
# escala log en Y: sin ella, 2006-2013 quedan aplastados frente al pico ~2022
ax.set_yscale("log")
ax.set_title("Lanzamientos por año (release)")
ax.set_xlabel("Año")
ax.set_ylabel("Cantidad de juegos")
ax.set_xticks(juegos_por_anio.index)
ax.tick_params(axis="x", rotation=45)
sns.despine()
plt.tight_layout()
plt.show()
"""
Analisis de la serie de lanzamientos por año:

* 2006-2012: oferta baja y relativamente estable (decenas a ~200 juegos/año).
* 2013-2014: salto fuerte (se cruza el orden de ~1000 lanzamientos).
* 2014-2018: crecimiento sostenido hasta varios miles por año.
* 2019: leve pausa/baja respecto a 2018; luego retoma alza en 2020-2021.
* 2022: maximo del periodo (~11000+ lanzamientos).
* 2023: cae a ~2500. No se interpreta como colapso del mercado: el scraping
  del dataset se tomo en un punto intermedio de 2023, no al cierre del año,
  asi que 2023 esta incompleto y no es comparable con años completos.

Uso analitico: esta curva da contexto de oferta temporal; al comparar reseñas
o tops por año hay que recordar que mas juegos lanzados != mas 'exito', y que
2023 no debe usarse como año de referencia plena.
"""

# ---------------------------------------------------------------------------
# 4) DISPERSION — total_reviews vs all_time_peak (escala log)
# ---------------------------------------------------------------------------
df_disp = df[(df["total_reviews"] > 0) & (df["all_time_peak"] > 0)]

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(
    df_disp["total_reviews"],
    df_disp["all_time_peak"],
    alpha=0.2,
    s=8,
    c="#264653",
    edgecolors="none",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("total_reviews (log)")
ax.set_ylabel("all_time_peak (log)")
ax.set_title("Dispersión: total_reviews vs all_time_peak")
sns.despine()
plt.tight_layout()
plt.show()
"""
Analisis de la dispersion (escala log-log, todos los juegos con valores > 0):

* La nube tiene forma alargada que sube hacia la derecha: asociacion positiva
  entre total_reviews y all_time_peak (a mas reseñas, tiende a haber mayor pico).
* La zona mas densa esta en valores bajos/medios (aprox. 10^1-10^3 reseñas y
  10^0-10^2 de peak): la mayoria del catalogo son juegos con volumen moderado.
* Hacia la esquina superior derecha hay menos puntos: pocos titulos concentran
  reseñas y picos muy altos (coherente con la cola derecha de P2).
* Hay dispersion vertical: para un mismo rango de reseñas el peak puede variar
  bastante; la relacion es de tendencia, no de igualdad exacta.
* Las lineas verticales/horizontales abajo a la izquierda vienen de conteos
  enteros (muchos juegos con el mismo 1, 2, 5...) vistos en escala log; no son
  un error del grafico.
* Muestra que reseñas y peak se mueven juntos (popularidad).
* No afirma causalidad ("las reviews causan el peak"); ambas crecen con la
  visibilidad del juego.
"""

# ---------------------------------------------------------------------------
# 5-8) BARRAS — top 15 publishers y developers (ciclo)
# ---------------------------------------------------------------------------
pub_agg = (
    df.groupby("publisher", observed=True)
    .agg(n_juegos=("game", "count"), sum_total_reviews=("total_reviews", "sum"))
)
dev_agg = (
    df.groupby("developer", observed=True)
    .agg(n_juegos=("game", "count"), sum_total_reviews=("total_reviews", "sum"))
)

rankings = [
    (pub_agg, "sum_total_reviews", "Top 15 publishers por suma de total_reviews", "Publisher", "teal"),
    (pub_agg, "n_juegos", "Top 15 publishers por cantidad de juegos", "Publisher", "teal"),
    (dev_agg, "sum_total_reviews", "Top 15 developers por suma de total_reviews", "Developer", "darkorange"),
    (dev_agg, "n_juegos", "Top 15 developers por cantidad de juegos", "Developer", "darkorange"),
]

for tabla, col_rank, titulo, etiqueta_y, color in rankings:
    top = tabla.nlargest(15, col_rank).sort_values(col_rank)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index.astype(str), top[col_rank], color=color)
    ax.set_title(titulo)
    ax.set_ylabel(etiqueta_y)
    # reseñas en millones (evitar 1e7); n_juegos se deja en enteros normales
    if col_rank == "sum_total_reviews":
        ax.set_xlabel("Suma de total_reviews (millones)")
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M")
        )
    else:
        ax.set_xlabel("Cantidad de juegos")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    sns.despine()
    plt.tight_layout()
    plt.show()
"""
Analisis de las barras (top 15) — popularidad vs productividad:

1) Publishers por suma de total_reviews (popularidad / atencion acumulada):
* Valve lidera de forma clara (~11.5M reseñas), muy por encima del resto.
* EA, Ubisoft, Rockstar, KRAFTON, etc. se agrupan en un segundo nivel (~1-3M).
* Del ~10 al 15 el descenso es mas suave (cerca de ~1M).
* Interpreta volumen de reseñas acumuladas del catalogo, no calidad ni n de juegos.

2) Publishers por cantidad de juegos (productividad de catalogo):
* Big Fish Games (~248) y 8floor (~222) encabezan; no aparecen como lideres
  del ranking por reseñas.
* Ubisoft, Square Enix, SEGA si cruzan ambos tops (visibles en reviews y en n).
* Del ~9 al 15 hay empate tecnico (~100-110 juegos).
* Mucha productividad (muchos titulos) no implica automaticamente el mayor
  volumen de reseñas (P2: popularidad != productividad).

3) Developers por suma de total_reviews:
* Valve otra vez domina (~10.5M+); KRAFTON, Facepunch, Rockstar North, CDPR
  siguen a distancia.
* Tras el salto de Valve, el top 2-15 baja de forma mas gradual (~2.2M a ~0.6M).
* Studios con pocos titulos muy grandes pueden concentrar muchas reseñas.

4) Developers por cantidad de juegos:
* Choice of Games, Creobit, Laush... lideran por volumen de titulos (~150-130).
* Casi no coinciden con el top de reseñas: alta cadencia de publicacion,
  menor acumulacion de reviews por estudio en este ranking.
* Square Enix aparece por n de juegos; Valve no esta en este top de productividad.

Puente entre las 4 figuras:
* Los rankings por suma de reviews destacan marcas/estudios de alto impacto
  (pocos hits, mucha atencion).
* Los rankings por n_juegos destacan catalogos prolíficos (casual, VN, indie
  de alto volumen).
* Comparar ambos criterios es el punto del analisis: no confundir 'quien publica
  mas' con 'quien acumula mas reseñas'.
"""

