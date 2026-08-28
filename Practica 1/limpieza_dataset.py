import pandas as pd
#cargar el dataset
df = pd.read_csv("dataset/game_data_all.csv" , encoding="utf-8")

#mostrar el número de filas y columnas del dataset
print("\nNúmero de filas y columnas:")
print(df.shape)

#mostrar las columnas del dataset
print("\nColumnas:")
print(df.columns)

#eliminar las columnas que no son relevantes
"""
* Unnamed: 0: índice
* link: enlace request a la página de Steam
* peak_players: redundante con "all_time_peak"
* total_reviews: es la suma de las reviews positivas y negativas
* rating: no es relevante
* store_asset_mod_time: fecha de modificación del asset en la tienda de Steam
* review_percentage: porcentaje de reviews positivas
* players_right_now: jugadores en línea en el momento de la consulta de creación del dataset
* 24_hour_peak: jugadores en línea en las últimas 24 horas en el momento de la consulta de creación del dataset
"""
df = df.drop(columns=[
    "Unnamed: 0",
    "link",
    "peak_players",
    "total_reviews",
    "rating",
    "store_asset_mod_time",
    "review_percentage",
    "players_right_now",
    "24_hour_peak",
])

#mostrar el número de columnas del dataset despues de eliminar las columnas no relevantes
print("\nNúmero de columnas despues de eliminar las no relevantes:")
print(df.shape[1])

#mostrar las columnas del dataset
print("\nColumnas despues de eliminar las no relevantes:")
print(df.columns)

#verrificar si hay valores duplicados
duplicados = df.duplicated().sum()
print("Registros duplicados:", duplicados) 
#no se encontraron duplicados

#verificar si hay valores nulos
nulos = df.isnull().sum()
print("Valores nulos:")
print(nulos)

#eliminar los valores nulos
df = df.dropna()

#mostrar el número de filas y columnas del dataset despues de eliminar los valores nulos
print("\nNúmero de filas despues de eliminar los valores nulos:")
print(df.shape[0])

#mostrar información del dataset
df.info()

#convertir release y all_time_peak_date a datetime
df["release"] = pd.to_datetime(df["release"])
df["all_time_peak_date"] = pd.to_datetime(df["all_time_peak_date"])

#mostrar información del dataset despues de convertir release y all_time_peak_date a datetime
df.info()
"""
Hay 59775 registros y 11 columnas
Las variables numericas son(3):
* positive_reviews: reviews positivas
* negative_reviews: reviews negativas
* all_time_peak: maximo de jugadores en línea
Las variables datetime son(2):
* release: fecha de lanzamiento del juego
* all_time_peak_date: fecha del maximo de jugadores en línea
Las variables string son(6):
* game: nombre del juego
* primary_genre: género principal del juego
* store_genres: géneros del juego
* publisher: editor del juego
* developer: desarrollador del juego
* detected_technologies: tecnologías detectadas en el juego
"""
#mostrar las primeras 5 filas del dataset
print(df.head())    

#verificar si hay valores nulos despues de convertir release y all_time_peak_date a datetime
nulos = df.isnull().sum()
print("Valores nulos despues de convertir release y all_time_peak_date a datetime:")
print(nulos)
#no hay valores nulos despues de convertir release y all_time_peak_date a datetime

#verificar si hay valores negativos en las variables numericas
print("Reviews positivas negativas:", 
      (df["positive_reviews"] < 0).sum())

print("Reviews negativas negativas:", 
      (df["negative_reviews"] < 0).sum())

print("Picos negativos:", 
      (df["all_time_peak"] < 0).sum())

#mostrar estadisticas descriptivas de las variables numericas
print(df[[
    "positive_reviews",
    "negative_reviews",
    "all_time_peak"
]].describe()) 
"""
No hay valores negativos en las variables numericas.
No se eliminaran valores que podrian considerarse atipicos.
Hay juegos con muchas/pocas reviews positivas y negativas. Es valido para el analisis.
"""

#mostrar la fecha minima y maxima de las variables datetime
print("Fecha minima de release:", df["release"].min())
print("Fecha maxima de release:", df["release"].max())
print("Fecha minima de all_time_peak_date:", df["all_time_peak_date"].min())
print("Fecha maxima de all_time_peak_date:", df["all_time_peak_date"].max())

#revisar si hay fechas de pico anteriores al lanzamiento
fechas_invalidas = df[
    df["all_time_peak_date"] < df["release"]
]
print("Fechas de pico anteriores al lanzamiento:", len(fechas_invalidas))

#mostrar las 20 primeras filas de las fechas de pico anteriores al lanzamiento
print(fechas_invalidas[
    ["game", "release", "all_time_peak_date", "all_time_peak"]
].head(20))
"""
Hay 4051 fechas de pico anteriores al lanzamiento.
La explicación es que hay juegos que se encontraban en early access cuando 
registraron su pico de jugadores antes de su lanzamiento oficial.
No se eliminaron estos registros porque no se consideran invalidos.
"""

# Valores únicos y frecuencia de variables categóricas
print("\n--- GAME ---")
print("Total de registros:", len(df))
print("Nombres de juegos únicos:", df["game"].nunique())
print("Nombres de juegos repetidos:", df["game"].duplicated().sum())
print("\nJuegos que aparecen más de una vez:")
print(df["game"].value_counts()[df["game"].value_counts() > 1])
print("\nNombres vacíos:")
print((df["game"].str.strip() == "").sum())

print("\n--- PRIMARY GENRE ---")
print("Valores únicos:", df["primary_genre"].nunique())
print(df["primary_genre"].value_counts())

print("\n--- STORE GENRES ---")
print("Combinaciones únicas:", df["store_genres"].nunique())
print(df["store_genres"].value_counts().head(20))

print("\n--- PUBLISHER ---")
print("Valores únicos:", df["publisher"].nunique())
print(df["publisher"].value_counts().head(20))

print("\n--- DEVELOPER ---")
print("Valores únicos:", df["developer"].nunique())
print(df["developer"].value_counts().head(20))

print("\n--- DETECTED TECHNOLOGIES ---")
print("Combinaciones únicas:", df["detected_technologies"].nunique())
print(df["detected_technologies"].value_counts().head(20))
"""
Se encontro 5 tipos diferentes "PRIMARY GENRE": "Unknown Genre(*)". 
Se reemplazara por "Unknown Genre" para que sea mas legible.
"""

#reemplazar el texto "Unknown Genre (*)" por "Unknown Genre"
df["primary_genre"] = df["primary_genre"].str.replace(
    r"Unknown Genre \(\d+\)", 
    "Unknown Genre", 
    regex=True
)

#mostrar los valores unicos y la frecuencia de las variables categoricas despues de reemplazar el texto "Unknown Genre (*)" por "Unknown Genre"
print("\n--- PRIMARY GENRE DESPUES DE REMPLAZAR EL TEXTO Unknown Genre (*) POR Unknown Genre ---")
print("Valores únicos:", df["primary_genre"].nunique())
print(df["primary_genre"].value_counts())

#mostrar los juegos con nombres repetidos
print("\n--- JUEGOS REPETIDOS ---")
juegos_repetidos = df["game"].value_counts()
juegos_repetidos = juegos_repetidos[juegos_repetidos > 1]
print("Cantidad de juegos con nombres repetidos:",
      len(juegos_repetidos))
print("\nPrimeros 20:")
print(juegos_repetidos.head(20))
"""
Hay 140 juegos con nombres repetidos. Se comprobo anteriormente que no hay filas duplicadas.
Estos son juegos diferentes pero con el mismo nombre. No se eliminaran
"""

# Verificación final del dataset limpio
print("\n--- VERIFICACIÓN FINAL ---")
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])
print("\nValores nulos:")
print(df.isnull().sum().sum())
print("\nDuplicados:")
print(df.duplicated().sum())
print("\nTipos de datos:")
print(df.dtypes)

# Guardar dataset limpio
df.to_csv("dataset/game_data_clean.csv", index=False, encoding="utf-8")
print("\nDataset limpio guardado correctamente.")