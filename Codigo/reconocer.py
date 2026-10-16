"""Reconocimiento de piezas en una foto (fase 6). Lo usan la aplicación y la evaluación de escenas.

Qué hace: para una foto, repite los pasos de las fases 2 a 4 con las mismas funciones:
lado mayor a 640 px, mediana 3x3, segmentación b1, recorte de 128x128 con máscara por pieza,
vector HOG + forma + LBP (8.139 columnas) y clase y probabilidad del modelo final.
Detalle en documentacion\\PlanesContextoDeFase\\Fase6_Contexto.md, sección 2.
Cómo se corre: no se corre; se importa. El modelo se carga una sola vez (cargar_modelo).
Qué escribe: nada.
"""
import json
import time

import cv2
import joblib
import numpy as np

from caracteristicas import car_forma, car_hog, car_lbp
from preprocesamiento import RAIZ, filtrar, normalizar, recorte_roi, segmentar

RUTA_MODELO = RAIZ / "modelos" / "modelo_final.joblib"
RUTA_INFO = RAIZ / "modelos" / "modelo_final.json"
RUTA_PARAMETROS = RAIZ / "resultados" / "fase2" / "parametros.json"
COLORES = {"Brick_1x4": (176, 111, 42), "Brick_2x2_L": (30, 138, 224),
           "Plate_2x2": (91, 154, 58), "Plate_2x4": (46, 58, 176)}  # BGR

_modelo = None


def cargar_modelo():
    """Carga el modelo final una sola vez y lo reutiliza."""
    global _modelo
    if _modelo is None:
        _modelo = joblib.load(RUTA_MODELO)
    return _modelo


def info_modelo():
    return json.loads(RUTA_INFO.read_text(encoding="utf-8"))


def parametros_fase2():
    """Método de segmentación y filtro elegidos en la fase 2 (los mismos de generar_roi.py)."""
    if RUTA_PARAMETROS.exists():
        p = json.loads(RUTA_PARAMETROS.read_text(encoding="utf-8"))
        return p["segmentacion"], p["filtro"]
    return "b1", "mediana_3"


def vector_pieza(roi_img, roi_masc):
    """Vector de 8.139 columnas, en el mismo orden que caracteristicas\\*.npz."""
    return np.concatenate([car_hog(roi_img), car_forma(roi_masc), car_lbp(roi_img, roi_masc)])


def detectar(img_bgr, modelo=None):
    """Devuelve (foto normalizada de 640 px, lista de piezas, milisegundos).
    Cada pieza: caja (x, y, w, h) en la foto normalizada, clase, confianza, probabilidades por
    clase, area_px y toca_borde."""
    t0 = time.perf_counter()
    modelo = modelo or cargar_modelo()
    metodo, filtro = parametros_fase2()
    img = normalizar(img_bgr)
    base = filtrar(img) if filtro == "mediana_3" else img
    piezas, vectores = [], []
    for p in segmentar(base, metodo):
        roi_img, roi_masc = recorte_roi(base, p["mascara"], p["caja"])
        if not roi_masc.any():  # pieza tan pequeña que desaparece al reducir el recorte
            continue
        vectores.append(vector_pieza(roi_img, roi_masc))
        piezas.append({"caja": tuple(int(v) for v in p["caja"]), "area_px": int(p["area_px"]),
                       "toca_borde": bool(p["toca_borde"])})
    if vectores:
        proba = modelo.predict_proba(np.stack(vectores))
        clases = list(modelo.classes_)
        for pieza, fila in zip(piezas, proba):
            k = int(np.argmax(fila))
            pieza["clase"] = clases[k]
            pieza["confianza"] = float(fila[k])
            pieza["probabilidades"] = {c: float(v) for c, v in zip(clases, fila)}
    return img, piezas, (time.perf_counter() - t0) * 1000.0


def dibujar(img, piezas):
    """Copia de la foto con un rectángulo y la etiqueta «clase – confianza %» por pieza."""
    salida = img.copy()
    for p in piezas:
        x, y, w, h = p["caja"]
        color = COLORES.get(p["clase"], (0, 0, 0))
        cv2.rectangle(salida, (x, y), (x + w, y + h), color, 2)
        texto = f"{p['clase']} {p['confianza'] * 100:.0f}%" + (" (borde)" if p["toca_borde"] else "")
        (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_txt = y - 6 if y - th - 8 >= 0 else y + h + th + 6
        cv2.rectangle(salida, (x, y_txt - th - 4), (x + tw + 6, y_txt + base - 2), color, cv2.FILLED)
        cv2.putText(salida, texto, (x + 3, y_txt - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return salida
