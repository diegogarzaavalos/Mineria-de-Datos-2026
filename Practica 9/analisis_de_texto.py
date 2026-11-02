import math
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

# ---------------------------------------------------------------------------
# PREPARACION: tokens de generos y tecnologias
# ---------------------------------------------------------------------------
df = pd.read_csv("dataset/game_data_clean.csv", encoding="utf-8")

"""
Practica 9 — Analisis de texto
Dos corpus del CSV (no hay descripciones largas):
1) store_genres: impacto/cobertura de cada genero en el catalogo.
2) detected_technologies: stack tecnico (Engine, SDK, etc.) para lectura
   de negocio; el color de la nube marca el prefijo de cada etiqueta.
"""

# store_genres viene como "Action (1), Indie (23)"; se quita el id entre parentesis.
def tokens_generos(serie: pd.Series) -> list[str]:
    out = []
    for valor in serie.fillna("").astype(str):
        for parte in valor.split(","):
            nombre = re.sub(r"\s*\(\d+\)\s*$", "", parte.strip())
            if nombre and nombre.lower() not in ("nan", "none"):
                out.append(nombre)
    return out


# detected_technologies: "Engine.Unity; SDK.cURL; ..."
def tokens_tecnologias(serie: pd.Series) -> list[str]:
    out = []
    for valor in serie.fillna("").astype(str):
        for parte in valor.split(";"):
            etiqueta = parte.strip()
            if etiqueta and etiqueta.lower() not in ("nan", "none"):
                out.append(etiqueta)
    return out


generos = tokens_generos(df["store_genres"])
tecnologias = tokens_tecnologias(df["detected_technologies"])
freq_generos = Counter(generos)
freq_tech = Counter(tecnologias)

# Prefijo antes del punto: Engine, SDK, AntiCheat, ...
def prefijo(etiqueta: str) -> str:
    return etiqueta.split(".", 1)[0] if "." in etiqueta else "OTRO"


freq_prefijos = Counter(prefijo(t) for t in tecnologias)

print("=== 1. EDA GENEROS (store_genres) ===")
print(f"Tokens totales: {len(generos)}")
print(f"Etiquetas unicas: {len(freq_generos)}")
print("Top 15:")
for nombre, n in freq_generos.most_common(15):
    print(f"  {nombre}: {n}")
"""
Indie, Action, Casual y Adventure dominan el catalogo: un juego suele
llevar varias etiquetas en store_genres, por eso la suma de tokens
supera el numero de filas.
"""

print("\n=== 2. EDA TECNOLOGIAS (detected_technologies) ===")
print(f"Tokens totales: {len(tecnologias)}")
print(f"Etiquetas unicas: {len(freq_tech)}")
print("Conteo por prefijo:")
for nombre, n in freq_prefijos.most_common():
    print(f"  {nombre}: {n}")
print("Top 15 etiquetas:")
for nombre, n in freq_tech.most_common(15):
    print(f"  {nombre}: {n}")
"""
Casi todo el volumen es SDK + Engine. El resto (Container, Emulator,
AntiCheat, Launcher) existe pero es raro. Unity lidera entre motores.
"""

# Color fijo por prefijo para leer la nube sin adivinar la categoria.
COLORES_PREFIJO = {
    "Engine": "#2a9d8f",
    "SDK": "#e9c46a",
    "AntiCheat": "#e76f51",
    "Emulator": "#264653",
    "Container": "#457b9d",
    "Launcher": "#9b5de5",
    "OTRO": "#6c757d",
}


def color_por_prefijo(word, font_size, position, orientation, random_state=None, **kwargs):
    # En la nube el texto puede ir sin prefijo; el color sigue la categoria.
    clave = prefijo_por_nombre.get(word, prefijo(word))
    return COLORES_PREFIJO.get(clave, COLORES_PREFIJO["OTRO"])


# Nombre corto en la nube (Unity, Vorbis…): no hay colisiones entre prefijos.
freq_tech_nube = Counter()
prefijo_por_nombre: dict[str, str] = {}
for etiqueta, n in freq_tech.items():
    nombre = etiqueta.split(".", 1)[1] if "." in etiqueta else etiqueta
    freq_tech_nube[nombre] += n
    prefijo_por_nombre[nombre] = prefijo(etiqueta)


# ---------------------------------------------------------------------------
# 3-4. FIGURAS SEPARADAS: nubes grandes + barras top en ventanas propias
# ---------------------------------------------------------------------------
# Juntas la nube queda aplastada; cada figura usa todo el ancho de pantalla.
sns.set_theme(style="whitegrid")

wc_generos = WordCloud(
    width=1600,
    height=900,
    background_color="white",
    colormap="viridis",
    prefer_horizontal=1.0,
    relative_scaling=0.45,
    min_font_size=14,
    max_font_size=220,
).generate_from_frequencies(freq_generos)

fig, ax = plt.subplots(figsize=(14, 8))
ax.imshow(wc_generos, interpolation="bilinear")
ax.axis("off")
ax.set_title("Nube de géneros (store_genres)", fontsize=14)
plt.tight_layout()
plt.show()

top_g = freq_generos.most_common(15)
nombres_g = [x[0] for x in top_g][::-1]
valores_g = [x[1] for x in top_g][::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(nombres_g, valores_g, color="#2a9d8f")
ax.set_title("Top 15 géneros")
ax.set_xlabel("Frecuencia (etiquetas en juegos)")
sns.despine(ax=ax)
plt.tight_layout()
plt.show()
"""
Nube = lectura visual; barra = orden y cantidades exactas.
Indie sigue siendo la etiqueta mas frecuente.
"""

# Nube completa (159): nombres sin prefijo; color = categoria.
# log1p: Unity sigue grande, pero las raras dejan de ser microscopicas
# (WordCloud ya las colocaba todas; en pantalla no se veian).
freq_tech_nube_viz = {k: math.log1p(v) for k, v in freq_tech_nube.items()}
wc_tech = WordCloud(
    width=2400,
    height=1400,
    background_color="white",
    prefer_horizontal=1.0,
    relative_scaling=0.5,
    min_font_size=11,
    max_font_size=120,
    max_words=len(freq_tech_nube_viz),
).generate_from_frequencies(freq_tech_nube_viz)
wc_tech.recolor(color_func=color_por_prefijo)

fig, ax = plt.subplots(figsize=(16, 10), dpi=120)
ax.imshow(wc_tech, interpolation="bilinear")
ax.axis("off")
ax.set_title(
    f"Nube de tecnologías ({len(wc_tech.words_)}/{len(freq_tech_nube)}; color = prefijo)",
    fontsize=14,
)
# Leyenda fuera del area de la nube para no tapar etiquetas.
for etiqueta, color in COLORES_PREFIJO.items():
    if etiqueta == "OTRO" or freq_prefijos.get(etiqueta, 0) == 0:
        continue
    ax.plot([], [], color=color, linewidth=6, label=etiqueta)
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    frameon=True,
    title="Prefijo",
)
plt.tight_layout()
plt.show()

top_t = freq_tech.most_common(15)
nombres_t = [x[0] for x in top_t][::-1]
valores_t = [x[1] for x in top_t][::-1]
colores_t = [COLORES_PREFIJO.get(prefijo(n), COLORES_PREFIJO["OTRO"]) for n in nombres_t]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(nombres_t, valores_t, color=colores_t)
ax.set_title("Top 15 tecnologías")
ax.set_xlabel("Frecuencia")
sns.despine(ax=ax)
plt.tight_layout()
plt.show()
"""
Verde = Engine, amarillo = SDK (resto segun leyenda).
Unity destaca entre motores; varios SDK acompañan mucho catalogo
sin definir el genero del juego.
"""

print("\n=== FIN Practica 9 ===")
