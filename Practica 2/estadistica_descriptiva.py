import pandas as pd

"""
Practica 2 — Estadistica descriptiva
Pregunta de analisis:
  Medir reseñas (positivas, negativas, total y proporcion positiva) a lo largo del
  tiempo segun el año de lanzamiento (release), segmentando por genero principal,
  y contrastar tops de publishers/developers por popularidad vs productividad.

Al final se actualiza game_data_clean.csv con total_reviews y
review_positive_percentage. No se guarda release_year (usar release.dt.year).
"""

# ---------------------------------------------------------------------------
# PREPARACION: dataset limpio + variables derivadas
# Se parte de game_data_clean.csv (Practica 1) para no repetir limpieza.
# ---------------------------------------------------------------------------
df = pd.read_csv("dataset/game_data_clean.csv", encoding="utf-8")

# En el CSV las fechas se guardan como texto; datetime permite .dt.year y rangos.
df["release"] = pd.to_datetime(df["release"])
df["all_time_peak_date"] = pd.to_datetime(df["all_time_peak_date"])

# total_reviews: base unica para rankings y agrupaciones (pos + neg).
df["total_reviews"] = df["positive_reviews"] + df["negative_reviews"]

# Proporcion positiva por juego: proxy de "recepcion", distinto del volumen.
# replace(0, NA) evita division por cero; mean/median ignoran NaN.
df["review_positive_percentage"] = df["positive_reviews"] / df["total_reviews"].replace(0, pd.NA)

# El año se obtiene en codigo con release.dt.year (sin columna extra en el df).
# Asi se conserva la fecha completa para practicas de series de tiempo.

# Solo primary_genre: un juego -> un genero (evita doble conteo de store_genres).
cols_numericas = [
    "positive_reviews",
    "negative_reviews",
    "total_reviews",
    "all_time_peak",
    "review_positive_percentage",
]

# ---------------------------------------------------------------------------
# 1) ESTADISTICA DESCRIPTIVA (numericas)
# Por que: conocer centro, dispersion y extremos antes de agrupar.
# describe() da media/std/cuartiles; la mediana se reporta aparte porque
# reseñas y picos tienen colas largas (outliers de titulos AAA/virales).
# ---------------------------------------------------------------------------
print("\n=== 1. ESTADISTICA DESCRIPTIVA (NUMERICAS) ===")
print(df[cols_numericas].describe().round(2))

print("\nMedianas (centro robusto ante outliers):")
print(df[cols_numericas].median().round(2))
"""
En reseñas y all_time_peak la media es mucho mayor que la mediana
(ej. total_reviews: media ~1608 vs mediana 27). Distribucion asimetrica
con cola a la derecha: pocos juegos extremos jalan la media hacia arriba.
La mediana describe mejor el juego tipico; por eso se usa en los groupby.
review_positive_percentage (media ~0.74, mediana ~0.80) esta acotada 0-1,
asi que los outliers distorsionan menos que en conteos de reseñas.
"""

# ---------------------------------------------------------------------------
# 2) FRECUENCIAS DE primary_genre
# Por que: saber que generos dominan el catalogo (tamanio de cada segmento).
# Sin esto, comparar sumas de reviews entre generos puede confundir volumen
# de catalogo con "exito" del genero.
# ---------------------------------------------------------------------------
print("\n=== 2. VARIABLE CATEGORICA: primary_genre ===")
freq_genero = df["primary_genre"].value_counts()
pct_genero = df["primary_genre"].value_counts(normalize=True) * 100
tabla_genero = pd.DataFrame({
    "frecuencia": freq_genero,
    "porcentaje": pct_genero.round(2),
})
print("Valores unicos:", df["primary_genre"].nunique())
print(tabla_genero)
"""
Indie y Action concentran gran parte del catalogo (~27% y ~22%).
Hay que tener ese tamanio en cuenta al interpretar sumas de reseñas por genero.
"""

# ---------------------------------------------------------------------------
# 3) LANZAMIENTOS POR AÑO
# Por que: el eje temporal no es uniforme; hace falta ver en que años hay
# mas oferta (n_juegos) antes de interpretar reseñas por año x genero.
# ---------------------------------------------------------------------------
print("\n=== 3. LANZAMIENTOS POR AÑO (release) ===")
# Serie temporal del año; no se guarda como columna del DataFrame.
anio_release = df["release"].dt.year
juegos_por_anio = df.groupby(anio_release).size().rename("n_juegos")
juegos_por_anio.index.name = "anio_release"
print(juegos_por_anio)
print("\nRango temporal:", df["release"].min().date(), "->", df["release"].max().date())
"""
Hay años con pocos lanzamientos y otros con muchos (sobre todo ~2014-2022).
No atribuir a "exito" lo que puede ser solo mayor oferta ese año.
Las reseñas son acumuladas al scrape; el tiempo es el año de release.
"""

# ---------------------------------------------------------------------------
# 4) AGRUPADO: año de release x primary_genre
# Por que: nucleo de la pregunta — como se distribuyen volumen de reseñas
# y valores tipicos (medianas) segun cuando salio el juego y su genero.
# - sum_*: volumen acumulado del grupo (popularidad agregada).
# - median_*: comportamiento tipico por juego (menos distorsionado por hits).
# - n_juegos: tamaño del grupo (contexto para no confundir suma con exito).
# ---------------------------------------------------------------------------
print("\n=== 4. AGRUPADO: año de release x primary_genre ===")
agrupado_anio_genero = (
    df.groupby([anio_release, "primary_genre"], observed=True)
    .agg(
        n_juegos=("game", "count"),
        sum_total_reviews=("total_reviews", "sum"),
        sum_positive=("positive_reviews", "sum"),
        sum_negative=("negative_reviews", "sum"),
        median_total_reviews=("total_reviews", "median"),
        median_positive_pct=("review_positive_percentage", "median"),
        median_all_time_peak=("all_time_peak", "median"),
    )
    .reset_index()
    .rename(columns={"release": "anio_release"})  # groupby nombra el nivel con el origen .dt
    .sort_values(["anio_release", "sum_total_reviews"], ascending=[True, False])
)
print(agrupado_anio_genero.head(40))
print("\nFilas totales del agrupado:", len(agrupado_anio_genero))
"""
El maximo de total_reviews llega a millones y el 75% de juegos tiene ~136 o menos:
las sumas las dominan pocos titulos; las medianas muestran el tipico del grupo.
"""

# ---------------------------------------------------------------------------
# 5) TOP PUBLISHERS — dos rankings a proposito
# Por que: "popularidad" y "productividad" no son lo mismo.
# - Por suma de total_reviews: quien acumula mas atencion/reseñas (popularidad).
# - Por n_juegos: quien publica mas titulos (productividad de catalogo).
# median_positive_pct se incluye como matiz: alto volumen != necesariamente
# buena recepcion (tambien puede haber base grande o polemica).
# Un publisher alto en juegos y bajo en reviews sugiere mucha productividad
# con poca popularidad relativa; el caso inverso sugiere catalogo pequeno
# pero muy visible.
# ---------------------------------------------------------------------------
print("\n=== 5. TOP PUBLISHERS ===")
top_pub_reviews = (
    df.groupby("publisher", observed=True)
    .agg(
        n_juegos=("game", "count"),
        sum_total_reviews=("total_reviews", "sum"),
        sum_positive=("positive_reviews", "sum"),
        sum_negative=("negative_reviews", "sum"),
        median_positive_pct=("review_positive_percentage", "median"),
    )
    .sort_values("sum_total_reviews", ascending=False)
    .head(15)
)
print("\n-- Top 15 por suma de total_reviews (popularidad / volumen de reseñas) --")
print(top_pub_reviews)

top_pub_juegos = (
    df.groupby("publisher", observed=True)
    .agg(
        n_juegos=("game", "count"),
        sum_total_reviews=("total_reviews", "sum"),
        median_positive_pct=("review_positive_percentage", "median"),
    )
    .sort_values("n_juegos", ascending=False)
    .head(15)
)
print("\n-- Top 15 por cantidad de juegos (productividad de catalogo) --")
print(top_pub_juegos)
"""
Los nombres del top por suma no coinciden del todo con el top por n_juegos:
confirma que popularidad acumulada y productividad de catalogo miden cosas distintas.
"""

# ---------------------------------------------------------------------------
# 6) TOP DEVELOPERS — misma logica que publishers
# Por que: developer y publisher son entidades distintas en el ERD.
# Un mismo estudio puede publicar poco y concentrar reseñas, o al reves.
# Comparar ambos rankings ayuda a separar exito de visibilidad vs ritmo
# de publicacion.
# ---------------------------------------------------------------------------
print("\n=== 6. TOP DEVELOPERS ===")
top_dev_reviews = (
    df.groupby("developer", observed=True)
    .agg(
        n_juegos=("game", "count"),
        sum_total_reviews=("total_reviews", "sum"),
        sum_positive=("positive_reviews", "sum"),
        sum_negative=("negative_reviews", "sum"),
        median_positive_pct=("review_positive_percentage", "median"),
    )
    .sort_values("sum_total_reviews", ascending=False)
    .head(15)
)
print("\n-- Top 15 por suma de total_reviews (popularidad / volumen de reseñas) --")
print(top_dev_reviews)

top_dev_juegos = (
    df.groupby("developer", observed=True)
    .agg(
        n_juegos=("game", "count"),
        sum_total_reviews=("total_reviews", "sum"),
        median_positive_pct=("review_positive_percentage", "median"),
    )
    .sort_values("n_juegos", ascending=False)
    .head(15)
)
print("\n-- Top 15 por cantidad de juegos (productividad de catalogo) --")
print(top_dev_juegos)
"""
Misma lectura que publishers: suma = popularidad; n_juegos = productividad.
"""

"""
Resumen para el reporte escrito:
1) Estadistica Descriptiva: caracteriza reseñas, pico y % positivo del dataset.
2) Frecuencias de primary_genre: tamaño de cada segmento de analisis.
3) Lanzamientos por año: contexto del eje temporal via release.dt.year.
4) Año x genero: responde como se concentran reseñas y tipicos por tiempo/genero.
5-6) Tops publisher/developer:
     - suma reviews = popularidad / atencion acumulada
     - n juegos = productividad
     - % positivo mediano = matiz de recepcion (no confundir con volumen)

Decisiones de diseño documentadas arriba en cada bloque.
"""

# Verificacion final antes de exportar
print("\n--- VERIFICACION FINAL ---")
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])
print("\nValores nulos:")
print(df.isnull().sum().sum())
print("\nDuplicados:")
print(df.duplicated().sum())
print("\nTipos de datos:")
print(df.dtypes)

# Guardar dataset actualizado
df.to_csv("dataset/game_data_clean.csv", index=False, encoding="utf-8")
print("\nDataset actualizado guardado correctamente.")
