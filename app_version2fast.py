import pandas as pd
import numpy as np
import xgboost as xgb
import sqlite3
import streamlit as st

# 1. Conectar a la base de datos SQLite
try:
    conn = sqlite3.connect(r'C:\Users\rafaf\Desktop\CODING\Mi Proyecto\archive\database.sqlite')
    st.write("✅ Conexión a la base de datos establecida correctamente.")
except sqlite3.Error as error:
    st.error(f"❌ Error al conectar a la base de datos: {error}")

# Seleccionar la tabla a explorar
tabla_a_explorar = 'matches'

# Cargar los datos desde la base de datos
datos = pd.read_sql_query(f"SELECT * FROM {tabla_a_explorar};", conn)

# Preprocesamiento de los datos
datos = datos.dropna(subset=['winner_name', 'loser_name'])

# Normalizar nombres de jugadores y manejar valores nulos
datos['winner_name'] = datos['winner_name'].fillna('').str.lower()
datos['loser_name'] = datos['loser_name'].fillna('').str.lower()

# Crear columnas jugador_1 y jugador_2 (orden alfabético)
datos[['jugador_1', 'jugador_2']] = np.sort(datos[['winner_name', 'loser_name']], axis=1)

# Crear columna 'ganador': 1 si jugador_1 gana, 0 si jugador_2 gana
datos['ganador'] = (datos['winner_name'] == datos['jugador_1']).astype(int)

# Asegurar que las columnas de ranking existen y reemplazar valores NaN
datos['winner_rank'] = datos['winner_rank'].fillna(999).astype(int)
datos['loser_rank'] = datos['loser_rank'].fillna(999).astype(int)

# Crear las columnas 'rank_jugador_1' y 'rank_jugador_2'
datos['rank_jugador_1'] = np.where(datos['winner_name'] == datos['jugador_1'], 
                                   datos['winner_rank'], datos['loser_rank'])

datos['rank_jugador_2'] = np.where(datos['winner_name'] == datos['jugador_1'], 
                                   datos['loser_rank'], datos['winner_rank'])

# Crear columna 'pareja' para identificar los enfrentamientos entre jugadores
datos['pareja'] = datos.apply(lambda x: tuple(sorted([x['winner_name'], x['loser_name']])), axis=1)

# Ordenar los datos por 'pareja' y 'tourney_date' para calcular el historial acumulativo
datos = datos.sort_values(by=['pareja', 'tourney_date'])

# Calcular el historial acumulativo de victorias para jugador_1 y jugador_2
datos['historial_ganador_1'] = datos.groupby('pareja')['ganador'].cumsum() - datos['ganador']
datos['historial_ganador_2'] = datos.groupby('pareja')['ganador'].cumcount() - datos['historial_ganador_1']

# Total de enfrentamientos previos entre los dos jugadores
datos['total_enfrentamientos'] = datos.groupby('pareja').cumcount()

# Porcentaje de victorias acumulado para jugador_1
datos['porcentaje_victorias_jugador_1'] = (datos['historial_ganador_1'] / datos['total_enfrentamientos']).fillna(0)

# Optimización: Crear nuevas características de forma vectorizada
cols = ['w_ace', 'l_ace', 'w_df', 'l_df', 'w_bpSaved', 'l_bpSaved', 'w_bpFaced', 'l_bpFaced', 'winner_age', 'loser_age']
for col in cols:
    datos[f"{col}_jugador_1"] = np.where(datos['winner_name'] == datos['jugador_1'], datos[col], datos[col.replace('w_', 'l_')])
    datos[f"{col}_jugador_2"] = np.where(datos['winner_name'] == datos['jugador_1'], datos[col.replace('w_', 'l_')], datos[col])

# Función para entrenar el modelo y predecir probabilidades
def entrenar_y_predecir(jugador_1, jugador_2, superficie, datos):
    jugador_1, jugador_2 = sorted([jugador_1.lower(), jugador_2.lower()])
    historial = datos[
        ((datos['jugador_1'] == jugador_1) & (datos['jugador_2'] == jugador_2)) |
        ((datos['jugador_1'] == jugador_2) & (datos['jugador_2'] == jugador_1))
    ]
    historial = historial[historial['surface'].str.lower() == superficie]

    if historial.empty:
        st.warning(f"⚠️ No hay datos históricos entre {jugador_1} y {jugador_2} en la superficie {superficie}.")
        return None

    # Cálculo de victorias
    victorias_jugador_1 = historial[historial['ganador'] == 1].shape[0]
    victorias_jugador_2 = historial[historial['ganador'] == 0].shape[0]

    # Total de enfrentamientos previos entre los dos jugadores
    total_enfrentamientos = historial.shape[0]

    # Mostrar el resultado de los enfrentamientos y victorias
    st.subheader(f"📊 Predicción en {superficie.capitalize()}")
    st.write(f"Total de enfrentamientos entre {jugador_1.capitalize()} y {jugador_2.capitalize()} en {superficie.capitalize()}: {total_enfrentamientos} partidos.")
    st.write(f"{jugador_1.capitalize()} ha ganado {victorias_jugador_1} veces.")
    st.write(f"{jugador_2.capitalize()} ha ganado {victorias_jugador_2} veces.")

    # Preparar las características para la predicción
    columnas_relevantes = [
        'rank_jugador_1', 'rank_jugador_2', 'historial_ganador_1',
        'historial_ganador_2', 'total_enfrentamientos', 'porcentaje_victorias_jugador_1',
        'w_ace_jugador_1', 'w_ace_jugador_2', 'w_df_jugador_1', 'w_df_jugador_2',
        'w_bpSaved_jugador_1', 'w_bpSaved_jugador_2', 'w_bpFaced_jugador_1',
        'w_bpFaced_jugador_2', 'winner_age_jugador_1', 'winner_age_jugador_2'
    ]

    X = historial[columnas_relevantes].copy()
    Y = historial['ganador'].copy()

    # Entrenar el modelo XGBoost
    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(X, Y)

    # Hacer una predicción para un partido nuevo entre ambos jugadores
    ultimo_partido = X.iloc[-1:].copy()
    probabilidad = model.predict_proba(ultimo_partido)[0]
    prob_jugador_1 = probabilidad[1] * 100
    prob_jugador_2 = probabilidad[0] * 100

    # Determinar quién tiene más chances de ganar
    ganador_predicho = jugador_1 if prob_jugador_1 > prob_jugador_2 else jugador_2

    # Mostrar la predicción
    st.write(f"🏆 **{ganador_predicho.capitalize()} tiene más chances de ganar**")
    st.write(f"➡️ **{jugador_1.capitalize()}: {prob_jugador_1:.2f}%**")
    st.write(f"➡️ **{jugador_2.capitalize()}: {prob_jugador_2:.2f}%**")

# Interfaz Streamlit
def interfaz():
    st.title("🎾 Predicción de Partido de Tenis")

    # Obtener la lista de jugadores únicos
    nombres_jugadores = sorted(pd.concat([datos['winner_name'], datos['loser_name']]).unique())

    # Seleccionar jugadores
    jugador_1 = st.selectbox("👤 Selecciona el primer jugador:", nombres_jugadores)
    jugador_2 = st.selectbox("👤 Selecciona el segundo jugador:", nombres_jugadores)

    if jugador_1 and jugador_2 and jugador_1 != jugador_2:
        if st.button("📊 Predecir resultados"):
            superficies = ['clay', 'grass', 'hard']
            for superficie in superficies:
                entrenar_y_predecir(jugador_1, jugador_2, superficie, datos)

if __name__ == "__main__":
    interfaz()
