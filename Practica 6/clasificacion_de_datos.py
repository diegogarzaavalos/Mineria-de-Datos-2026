import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

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
Practica 6 — Clasificacion con KNN
Pregunta: ¿se puede predecir el genero principal (top 8) a partir de
volumen de reseñas, pico de jugadores y % positivo?
Y = primary_genre (top 8, misma seleccion que P3 y P4).
X = log10(total_reviews), log10(all_time_peak), review_positive_percentage.
El genero no entra como predictor: seria la propia etiqueta.
"""

# log exige valores > 0; el % positivo no puede ser nulo
df_clf = df[
    (df["total_reviews"] > 0) & (df["all_time_peak"] > 0)
].dropna(subset=["review_positive_percentage"]).copy()

top8 = df_clf["primary_genre"].value_counts().head(8).index.tolist()
df_clf = df_clf[df_clf["primary_genre"].isin(top8)].copy()

df_clf["log_total_reviews"] = np.log10(df_clf["total_reviews"])
df_clf["log_all_time_peak"] = np.log10(df_clf["all_time_peak"])

features = [
    "log_total_reviews",
    "log_all_time_peak",
    "review_positive_percentage",
]
X = df_clf[features]
y = df_clf["primary_genre"]

print("=== 1. DATOS PARA CLASIFICAR ===")
print("Clases (top 8):", top8)
print("Filas:", len(df_clf))
print("\nJuegos por genero:")
print(y.value_counts().reindex(top8))
"""
Se limita al top 8 para no clasificar generos con muy pocos juegos.
Indie y Action son mucho mas frecuentes: la accuracy puede inflarse
si el modelo solo acierta esas clases. La matriz de confusion lo muestra.
El log en reseñas y pico evita que los hits dominen la distancia de KNN.
"""

# ---------------------------------------------------------------------------
# 2. TRAIN / TEST Y ESCALADO
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# El escalador se ajusta solo con entrenamiento para no filtrar info del test
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("\n=== 2. PARTICION ===")
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))
"""
stratify conserva la proporcion de generos en ambos conjuntos.
StandardScaler: KNN usa distancia. Sin escalar, log_reviews (unidades de
log) y el % positivo (0 a 1) no pesan igual. Se ajusta el scaler con train
y solo se transforma el test.
"""

# ---------------------------------------------------------------------------
# 3. CICLO DE K
# ---------------------------------------------------------------------------
print("\n=== 3. ACCURACY POR K (conjunto de prueba) ===")
ks = [5, 11, 21]
resultados_k = []
for k in ks:
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_train_s, y_train)
    pred = modelo.predict(X_test_s)
    acc = accuracy_score(y_test, pred)
    resultados_k.append((k, acc))
    print(f"k = {k:2d}  accuracy = {acc:.4f}")

k_elegido, acc_elegida = max(resultados_k, key=lambda par: par[1])
print(f"\nk elegido = {k_elegido} (accuracy = {acc_elegida:.4f})")
"""
k chico sigue mas el ruido local; k grande suaviza y se acerca a la clase
mas frecuente. Se comparan 5, 11 y 21 en el conjunto de prueba y se queda
el de mayor accuracy. Es una busqueda simple, no una validacion cruzada.
"""

# ---------------------------------------------------------------------------
# 4. MODELO FINAL Y MATRIZ DE CONFUSION
# ---------------------------------------------------------------------------
modelo_final = KNeighborsClassifier(n_neighbors=k_elegido)
modelo_final.fit(X_train_s, y_train)
y_pred = modelo_final.predict(X_test_s)

print("\n=== 4. EVALUACION DEL MODELO FINAL ===")
print(f"Accuracy en prueba: {accuracy_score(y_test, y_pred):.4f}")

# orden fijo = top 8, para leer filas y columnas igual
matriz = confusion_matrix(y_test, y_pred, labels=top8)
tabla_matriz = pd.DataFrame(matriz, index=top8, columns=top8)
print("\nMatriz de confusion (filas = real, columnas = predicho):")
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)
print(tabla_matriz)

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    tabla_matriz,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax,
)
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
ax.set_title(f"Matriz de confusion KNN (k = {k_elegido})")
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
"""
Accuracy en prueba ≈ 0.26 con k = 21 (mejor de 5, 11 y 21).
La clase mas frecuente, Indie, es ≈ 28% del test: predecir siempre Indie
supera a este KNN. Subir k mejoro un poco la accuracy, pero no basta.

Matriz (filas reales):
* Casi todas las filas concentran predicciones en Indie y Action.
* Diagonal debil fuera de esas dos: Adventure ~110, Casual ~131,
  Strategy ~30, Simulation ~18, RPG ~17, Free to Play ~2 aciertos.
* Free to Play y Simulation casi no se recuperan: el modelo los manda
  a Indie o Action.

Lectura metodologica: volumen de reseñas, pico y % positivo no separan
bien el genero principal. Coherente con P4, donde las medianas de %
positivo entre generos estaban cercanas (~0.75-0.82). No se concluye
nada de 'calidad' del genero; solo que estas tres X no alcanzan para
clasificar la etiqueta.
"""
