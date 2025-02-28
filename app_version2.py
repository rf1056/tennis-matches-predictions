import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import sqlite3
import os
import streamlit as st
import plotly.express as px
from PIL import Image

# 1. Conectar a la base de datos SQLite
try:
    conn = sqlite3.connect(#UPDATE DATA LOCATION#)
    st.write("Conexión a la base de datos establecida correctamente.")
except sqlite3.Error as error:
    st.error(f"Error al conectar a la base de datos: {error}")

# Seleccionar la tabla a explorar
tabla_a_explorar = 'matches'

# Cargar los datos desde la base de datos
datos = pd.read_sql_query(f"SELECT * FROM {tabla_a_explorar};", conn)

# Preprocesamiento de los datos
datos = datos.dropna(subset=['winner_name', 'loser_name'])

# Normalizar nombres de jugadores para evitar problemas de mayúsculas/minúsculas
datos['winner_name'] = datos['winner_name'].str.lower()
datos['loser_name'] = datos['loser_name'].str.lower()

# Crear columnas jugador_1 y jugador_2 (orden alfabético)
datos['jugador_1'] = datos.apply(lambda x: sorted([x['winner_name'], x['loser_name']])[0], axis=1)
datos['jugador_2'] = datos.apply(lambda x: sorted([x['winner_name'], x['loser_name']])[1], axis=1)

# Crear columna 'ganador': 1 si jugador_1 gana, 0 si jugador_2 gana
datos['ganador'] = datos.apply(lambda x: 1 if x['winner_name'] == x['jugador_1'] else 0, axis=1)

# Asignar ranking correspondiente
datos['rank_jugador_1'] = datos.apply(lambda x: x['winner_rank'] if x['winner_name'] == x['jugador_1'] else x['loser_rank'], axis=1)
datos['rank_jugador_2'] = datos.apply(lambda x: x['loser_rank'] if x['winner_name'] == x['jugador_1'] else x['winner_rank'], axis=1)

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
datos['porcentaje_victorias_jugador_1'] = datos['historial_ganador_1'] / datos['total_enfrentamientos']
datos['porcentaje_victorias_jugador_1'] = datos['porcentaje_victorias_jugador_1'].fillna(0)  # Manejar divisiones por cero

# Nuevas características relevantes para el modelo
datos['aces_jugador_1'] = datos.apply(lambda x: x['w_ace'] if x['winner_name'] == x['jugador_1'] else x['l_ace'], axis=1)
datos['aces_jugador_2'] = datos.apply(lambda x: x['l_ace'] if x['winner_name'] == x['jugador_1'] else x['w_ace'], axis=1)

datos['df_jugador_1'] = datos.apply(lambda x: x['w_df'] if x['winner_name'] == x['jugador_1'] else x['l_df'], axis=1)
datos['df_jugador_2'] = datos.apply(lambda x: x['l_df'] if x['winner_name'] == x['jugador_1'] else x['w_df'], axis=1)

datos['bpsaved_jugador_1'] = datos.apply(lambda x: x['w_bpSaved'] if x['winner_name'] == x['jugador_1'] else x['l_bpSaved'], axis=1)
datos['bpsaved_jugador_2'] = datos.apply(lambda x: x['l_bpSaved'] if x['winner_name'] == x['jugador_1'] else x['w_bpSaved'], axis=1)

datos['bpFaced_jugador_1'] = datos.apply(lambda x: x['w_bpFaced'] if x['winner_name'] == x['jugador_1'] else x['l_bpFaced'], axis=1)
datos['bpFaced_jugador_2'] = datos.apply(lambda x: x['l_bpFaced'] if x['winner_name'] == x['jugador_1'] else x['w_bpFaced'], axis=1)

datos['edad_jugador_1'] = datos.apply(lambda x: x['winner_age'] if x['winner_name'] == x['jugador_1'] else x['loser_age'], axis=1)
datos['edad_jugador_2'] = datos.apply(lambda x: x['loser_age'] if x['winner_name'] == x['jugador_1'] else x['winner_age'], axis=1)

# Función para mostrar la imagen del jugador
def mostrar_imagen_jugador(jugador):
    imagenes = {
        'roger federer': r'C:\Users\rafaf\Desktop\Mi Proyecto\imagenes_jugadores\federer.jpg',
        'rafael nadal': r'C:\Users\rafaf\Desktop\Mi Proyecto\imagenes_jugadores\nadal.jpg',
        'novak djokovic': r'C:\Users\rafaf\Desktop\Mi Proyecto\imagenes_jugadores\djokovic.jpg'
    }
    
    jugador = jugador.lower()
    if jugador in imagenes:
        imagen = Image.open(imagenes[jugador])
        st.image(imagen, caption=jugador.capitalize(), use_container_width=True)
    else:
        st.warning(f"No se encontró imagen para {jugador.capitalize()}.")

# Función para entrenar el modelo
def entrenar_modelo_por_pareja_y_superficie(jugador_1, jugador_2, superficie, datos):
    jugador_1, jugador_2 = sorted([jugador_1.lower(), jugador_2.lower()])
    historial = datos[
        ((datos['jugador_1'] == jugador_1) & (datos['jugador_2'] == jugador_2)) |
        ((datos['jugador_1'] == jugador_2) & (datos['jugador_2'] == jugador_1))
    ]
    historial = historial[historial['surface'].str.lower() == superficie]

    if historial.empty:
        st.warning(f"No hay datos históricos entre {jugador_1} y {jugador_2} en la superficie {superficie}.")
        return None, None, None

    columnas_relevantes = [
        'rank_jugador_1', 'rank_jugador_2', 'historial_ganador_1',
        'historial_ganador_2', 'total_enfrentamientos', 'porcentaje_victorias_jugador_1',
        'aces_jugador_1', 'aces_jugador_2', 'df_jugador_1', 'df_jugador_2',
        'bpsaved_jugador_1', 'bpsaved_jugador_2', 'bpFaced_jugador_1',
        'bpFaced_jugador_2', 'edad_jugador_1', 'edad_jugador_2'
    ]

    X = historial[columnas_relevantes].copy()
    Y = historial['ganador'].copy()

    scale_pos_weight = len(Y) / (2 * sum(Y == 1)) if sum(Y == 1) != 0 else 1.5
    model = xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=scale_pos_weight)
    model.fit(X, Y)

    archivo_modelo = f'modelo_{jugador_1}_{jugador_2}_{superficie}.joblib'
    joblib.dump({'modelo': model, 'columnas': X.columns.tolist()}, archivo_modelo)
    st.write(f"Modelo entrenado y guardado como {archivo_modelo}.")
    return model, X.columns.tolist(), archivo_modelo


# Función para predecir el ganador con la probabilidad
def predecir_y_mostrar_grafico(jugador_1, jugador_2, superficie, archivo_modelo, datos):
    if not os.path.exists(archivo_modelo):
        st.error(f"El archivo del modelo '{archivo_modelo}' no existe. Asegúrate de entrenar el modelo primero.")
        return None

    # Cargar el modelo
    modelo_guardado = joblib.load(archivo_modelo)
    modelo = modelo_guardado['modelo']
    columnas_esperadas = modelo_guardado['columnas']

    # Normalizar nombres de jugadores
    jugador_1, jugador_2 = sorted([jugador_1.lower(), jugador_2.lower()])

    # Filtrar el historial entre los jugadores
    historial = datos[
        ((datos['jugador_1'] == jugador_1) & (datos['jugador_2'] == jugador_2)) |
        ((datos['jugador_1'] == jugador_2) & (datos['jugador_2'] == jugador_1))
    ]
    
    if historial.empty:
        st.error(f"No hay historial previo entre {jugador_1} y {jugador_2}.")
        return None

    # Tomar los datos más recientes para la predicción
    ultimo_partido = historial.iloc[-1]
    X_nuevo = pd.DataFrame([{col: ultimo_partido[col] for col in columnas_esperadas}])

    # Hacer la predicción y obtener la probabilidad
    prediccion = modelo.predict(X_nuevo)[0]
    probabilidad = modelo.predict_proba(X_nuevo)[0]

    # Crear un gráfico de pastel para la superficie actual
    fig = px.pie(
        values=[probabilidad[1], probabilidad[0]],
        names=[f"{jugador_1.capitalize()} gana", f"{jugador_2.capitalize()} gana"],
        title=f"Probabilidades en {superficie.capitalize()}",
        color_discrete_sequence=['blue','red']
    )
    st.plotly_chart(fig)


# Interfaz Streamlit
def interfaz():
    st.title("Predicción de Partido de Tenis by Mayor 2")

    # Extraer nombres únicos de jugadores
    nombres_jugadores = sorted(pd.concat([datos['winner_name'], datos['loser_name']]).unique())

    # Dropdowns para seleccionar los jugadores
    jugador_1 = st.selectbox("Selecciona el primer jugador:", nombres_jugadores)
    jugador_2 = st.selectbox("Selecciona el segundo jugador:", nombres_jugadores)

    if jugador_1 and jugador_2 and jugador_1 != jugador_2:
        # Mostrar imágenes de los jugadores seleccionados
        st.subheader(f"Imagen de {jugador_1.capitalize()}")
        mostrar_imagen_jugador(jugador_1)
        
        st.subheader(f"Imagen de {jugador_2.capitalize()}")
        mostrar_imagen_jugador(jugador_2)

        if st.button("Entrenar y predecir para todas las superficies"):
            superficies = ['clay', 'grass', 'hard']
            for superficie in superficies:
                st.write(f"\nEntrenando y prediciendo para la superficie: {superficie.capitalize()}")
                modelo, columnas, archivo_modelo = entrenar_modelo_por_pareja_y_superficie(jugador_1, jugador_2, superficie, datos)
                if modelo:
                    predecir_y_mostrar_grafico(jugador_1, jugador_2, superficie, archivo_modelo, datos)
                else:
                    st.warning("El modelo no pudo ser entrenado.")

        st.markdown("""<p class='dato_curioso'>Dato curioso: ¿Sabías que el tenis moderno se originó en Birmingham, Inglaterra, en 1873? y sabias que el deporte mas quemante del mundo?""", unsafe_allow_html=True)
    else:
        st.warning("Por favor, ingrese dos jugadores diferentes.")

if __name__ == "__main__":
    interfaz()
