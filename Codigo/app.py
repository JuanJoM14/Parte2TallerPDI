"""Aplicación de la fase 6: reconoce las piezas Lego de una foto.

Qué hace: abre una foto (JPG o PNG), encuentra todas las piezas, dibuja su rectángulo con la
clase y la confianza, cuenta las piezas por clase y muestra el tiempo de procesamiento.
Usa reconocer.py (los mismos pasos de las fases 2 a 4) y el modelo final ya entrenado.
Cómo se corre (desde Taller 2):  .venv\\Scripts\\streamlit run Codigo\\app.py
Qué escribe: nada.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import streamlit as st

from reconocer import cargar_modelo, detectar, dibujar, info_modelo

NOMBRES = {"Brick_1x4": "Brick 1x4", "Brick_2x2_L": "Brick 2x2 L", "Plate_2x2": "Plate 2x2", "Plate_2x4": "Plate 2x4"}


@st.cache_resource
def modelo_en_memoria():
    return cargar_modelo()


def main():
    st.set_page_config(page_title="Clasificador de piezas Lego", layout="wide")
    st.title("Clasificador de piezas Lego")
    info = info_modelo()
    st.caption(f"Modelo: {info['experimento']} ({info['descriptores']}), {info['clasificador']}. "
               "Clases: Brick 1x4, Brick 2x2 L, Plate 2x2 y Plate 2x4.")

    archivo = st.file_uploader("Abrir una imagen", type=["jpg", "jpeg", "png"])
    if archivo is None:
        st.info("Sube una foto con una o varias piezas sobre fondo claro.")
        return
    img = cv2.imdecode(np.frombuffer(archivo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("No se pudo leer la imagen.")
        return

    foto, piezas, ms = detectar(img, modelo_en_memoria())
    anotada = dibujar(foto, piezas)

    col_img, col_res = st.columns([3, 2])
    with col_img:
        st.image(cv2.cvtColor(anotada, cv2.COLOR_BGR2RGB))
    with col_res:
        st.metric("Piezas detectadas", len(piezas))
        st.metric("Tiempo de procesamiento", f"{ms:.0f} ms")
        st.subheader("Conteo por clase")
        conteo = {NOMBRES[c]: sum(p["clase"] == c for p in piezas) for c in NOMBRES}
        st.table({"Clase": list(conteo.keys()), "Piezas": list(conteo.values())})

    if piezas:
        st.subheader("Detalle por pieza")
        st.table({"N.º": list(range(1, len(piezas) + 1)),
                  "Clase": [NOMBRES[p["clase"]] for p in piezas],
                  "Confianza": [f"{p['confianza'] * 100:.1f} %" for p in piezas],
                  "Área (px)": [p["area_px"] for p in piezas],
                  "Toca el borde": ["sí" if p["toca_borde"] else "no" for p in piezas]})
        if any(p["toca_borde"] for p in piezas):
            st.warning("Hay piezas cortadas por el borde de la foto. El modelo no se entrenó con piezas "
                       "cortadas, así que su resultado es menos confiable.")
    else:
        st.warning("No se encontraron piezas en la imagen.")


main()
