"""Estudio de parámetros de HOG (fase 3, lo pide la rúbrica).

Qué hace: calcula HOG con varias configuraciones sobre los recortes de roi\\ y, para cada una,
entrena un mismo clasificador fijo (estandarización + SVM lineal, C=1) con entrenamiento y lo
mide en validación (accuracy y F1 macro). La prueba no se usa.
La configuración del vector final no cambia: 9 orientaciones, celdas 8x8, bloques 2x2, como pide el
enunciado. El estudio solo muestra cómo cambian la dimensión y el desempeño con los parámetros.
Cómo se corre (desde Taller 2):  .venv\\Scripts\\python Codigo\\estudio_hog.py
Qué escribe: resultados\\fase3\\estudio_hog.csv. Se detiene si ya existe.
"""
import time

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocesamiento import RAIZ, escribir_csv, leer_csv, leer_png

DIR_ROI = RAIZ / "roi"
DIR_RES = RAIZ / "resultados" / "fase3"
SEMILLA = 42
# (orientaciones, píxeles por celda, celdas por bloque); la primera es la del enunciado
CONFIGURACIONES = [(9, 8, 2), (6, 8, 2), (12, 8, 2), (9, 16, 2), (9, 4, 2), (9, 8, 3)]


def vectores(grises, ori, celda, bloque):
    return np.stack([hog(g, orientations=ori, pixels_per_cell=(celda, celda),
                         cells_per_block=(bloque, bloque), block_norm="L2-Hys",
                         feature_vector=True) for g in grises]).astype(np.float32)


def main():
    salida = DIR_RES / "estudio_hog.csv"
    if salida.exists():
        raise SystemExit("resultados\\fase3\\estudio_hog.csv ya existe; moverlo para repetir.")
    filas = leer_csv(DIR_ROI / "metadatos.csv")
    datos = {}
    for uso in ("train", "val"):
        sub = [f for f in filas if f["uso"] == uso]
        grises = [cv2.cvtColor(leer_png(DIR_ROI / "imagenes" / f["nombre_roi"]), cv2.COLOR_BGR2GRAY)
                  for f in sub]
        datos[uso] = (grises, np.array([f["clase"] for f in sub]))
    resultados = []
    for ori, celda, bloque in CONFIGURACIONES:
        t0 = time.perf_counter()
        x_tr = vectores(datos["train"][0], ori, celda, bloque)
        x_va = vectores(datos["val"][0], ori, celda, bloque)
        modelo = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0, random_state=SEMILLA))
        modelo.fit(x_tr, datos["train"][1])
        pred = modelo.predict(x_va)
        y_va = datos["val"][1]
        resultados.append({
            "orientaciones": ori, "celda_px": f"{celda}x{celda}", "bloque_celdas": f"{bloque}x{bloque}",
            "dimension": x_tr.shape[1],
            "accuracy_val": round(accuracy_score(y_va, pred), 4),
            "f1_macro_val": round(f1_score(y_va, pred, average="macro"), 4),
            "segundos": round(time.perf_counter() - t0, 1),
            "es_la_del_enunciado": int((ori, celda, bloque) == (9, 8, 2))})
        print(resultados[-1])
    DIR_RES.mkdir(parents=True, exist_ok=True)
    escribir_csv(salida, list(resultados[0].keys()), resultados)


if __name__ == "__main__":
    main()
