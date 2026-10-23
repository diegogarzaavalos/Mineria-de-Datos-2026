import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# PREPARACION
# ---------------------------------------------------------------------------
df = pd.read_csv("dataset/game_data_clean.csv", encoding="utf-8")

if "total_reviews" not in df.columns:
    df["total_reviews"] = df["positive_reviews"] + df["negative_reviews"]
if "review_positive_percentage" not in df.columns:
    df["review_positive_percentage"] = (
        df["positive_reviews"] / df["total_reviews"].replace(0, pd.NA)
    )

"""
Practica 7 — Agrupamiento con K-Means
Pregunta: ¿los juegos se agrupan por volumen de reseñas, pico y % positivo,
sin usar el genero como etiqueta?
X = log10(total_reviews), log10(all_time_peak), review_positive_percentage.
El genero no entra al modelo (K-Means no es supervisado). Se cruza al final
solo para ver si los grupos coinciden con primary_genre.
"""

df_km = df[
    (df["total_reviews"] > 0) & (df["all_time_peak"] > 0)
].dropna(subset=["review_positive_percentage"]).copy()

df_km["log_total_reviews"] = np.log10(df_km["total_reviews"])
df_km["log_all_time_peak"] = np.log10(df_km["all_time_peak"])

features = [
    "log_total_reviews",
    "log_all_time_peak",
    "review_positive_percentage",
]
X = df_km[features].to_numpy(dtype=float)

# K-Means usa distancia: sin escalar, una variable domina los centroides
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

print("=== 1. DATOS ===")
print("Filas:", len(df_km))
print("Variables:", features)
"""
Misma base numerica que el KNN de P6, para poder comparar: alla el genero
no se separaba bien; aqui se pregunta si esas variables forman grupos propios.
"""

# Silhouette en todo el catalogo es cuadratico; se estima en una muestra fija
rng = np.random.default_rng(42)
idx_silueta = rng.choice(len(X_s), size=8000, replace=False)

# ---------------------------------------------------------------------------
# 2. ELECCION DE K (inercia y silhouette)
# ---------------------------------------------------------------------------
print("\n=== 2. BUSQUEDA DE K ===")
ks = range(2, 9)
inercias = []
siluetas = []
for k in ks:
    modelo_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    etiquetas_k = modelo_k.fit_predict(X_s)
    silueta_k = silhouette_score(X_s[idx_silueta], etiquetas_k[idx_silueta])
    inercias.append(modelo_k.inertia_)
    siluetas.append(silueta_k)
    print(f"k = {k}  inercia = {modelo_k.inertia_:.1f}  silhouette = {silueta_k:.4f}")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(ks), inercias, marker="o", color="#2a9d8f")
axes[0].set_title("Inercia (codo)")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inercia")
axes[1].plot(list(ks), siluetas, marker="o", color="#e76f51")
axes[1].set_title("Silhouette (muestra de 8000)")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette")
plt.tight_layout()
plt.show()
"""
Inercia: 2→3 es la caida mas grande (≈101817 a ≈65441); despues el
descenso se aplana.
Silhouette: maximo en k=2 (≈0.42) y k=3 casi igual (≈0.41); de k=4 en
adelante baja a ≈0.32-0.36.
Se elige k=3: separacion parecida a k=2 y un grupo adicional donde la
inercia todavia cae fuerte. k>3 parte de mas sin mejorar la silueta.
"""

# ---------------------------------------------------------------------------
# 3. K-MEANS FINAL
# ---------------------------------------------------------------------------
K_ELEGIDO = 3
modelo = KMeans(n_clusters=K_ELEGIDO, random_state=42, n_init=10)
df_km["cluster"] = modelo.fit_predict(X_s)

centroides_log = scaler.inverse_transform(modelo.cluster_centers_)
centroides = pd.DataFrame(
    centroides_log,
    columns=["log10_total_reviews", "log10_all_time_peak", "review_positive_percentage"],
)
centroides["total_reviews_aprox"] = 10 ** centroides["log10_total_reviews"]
centroides["all_time_peak_aprox"] = 10 ** centroides["log10_all_time_peak"]
centroides.insert(0, "cluster", centroides.index)
tamanos = df_km["cluster"].value_counts().sort_index()

print("\n=== 3. MODELO FINAL k = 3 ===")
print("Tamaño de cada grupo:")
print(tamanos.to_string())
print("\nCentroides (log y valor aproximado en escala original):")
print(
    centroides[
        [
            "cluster",
            "log10_total_reviews",
            "total_reviews_aprox",
            "log10_all_time_peak",
            "all_time_peak_aprox",
            "review_positive_percentage",
        ]
    ]
    .round(3)
    .to_string(index=False)
)
"""
Los centroides de reseñas y pico se muestran tambien como 10**log,
para leerlos en la escala del juego y no solo en log.
Tres perfiles numericos (orden de cluster puede cambiar si cambia la semilla;
con random_state=42 queda fijo):
* Grupo 0 (≈30798): pocos reviews (~15) y pico (~5), % positivo alto (~0.86).
* Grupo 1 (≈12688): pocos reviews (~11) y pico (~4), % positivo bajo (~0.39).
* Grupo 2 (≈16289): mucho mas volumen (~621 reviews, pico ~322) y % positivo ~0.78.
El genero no definio estos grupos: salen de volumen y recepcion.
"""

# ---------------------------------------------------------------------------
# 4. CRUCE CON primary_genre (no se uso para entrenar)
# ---------------------------------------------------------------------------
top8 = df_km["primary_genre"].value_counts().head(8).index
cruce = pd.crosstab(
    df_km.loc[df_km["primary_genre"].isin(top8), "cluster"],
    df_km.loc[df_km["primary_genre"].isin(top8), "primary_genre"],
).reindex(columns=top8)

print("\n=== 4. CRUCE GRUPO x TOP 8 GENEROS (solo lectura) ===")
print(cruce)
"""
Cada genero del top 8 aparece en los tres grupos. No hay un cluster
equivalente a un genero. Coherente con P6: estas tres variables no
separan primary_genre; K-Means agrupa por volumen de reseñas, pico
y % positivo.
"""

