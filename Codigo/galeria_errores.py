"""Galería de errores en prueba de los modelos distintos al final (fase 5, complemento).

Por qué existe: el modelo final (E4, SVM lineal) y el mejor de E1 no fallaron ninguna pieza en prueba,
así que sus galerías no muestran errores. Esta galería muestra las piezas de prueba que sí fallaron
otros modelos, para analizar la causa probable (enunciado 3.5).
Qué hace: lee resultados\\fase5\\predicciones_prueba.csv (ya calculado por evaluar.py) y dibuja cada pieza
mal clasificada, sin repetir, con su clase real, la clase predicha y la probabilidad del modelo que se
equivocó con más seguridad, y en cuántos modelos falla. Deja fuera E3 (LBP solo, 18 a 28 errores por
modelo), que llenaría la galería de casos poco informativos; sus errores están en las matrices de E3.
No carga ningún modelo ni vuelve a evaluar la prueba.
Cómo se corre (desde Taller 2):  .venv\\Scripts\\python Codigo\\galeria_errores.py
Qué escribe: resultados\\fase5\\galeria_errores_otros_modelos.png y galeria_errores_otros_modelos.csv
(una fila por pieza y modelo que la falla). Se detiene si el PNG ya existe.
"""
from collections import defaultdict

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from preprocesamiento import RAIZ, escribir_csv, leer_csv, leer_png

DIR_RES = RAIZ / "resultados" / "fase5"
DIR_ROI = RAIZ / "roi" / "imagenes"
EXCLUIR = ("E3",)
NOMBRES_CLF = {"svm_lineal": "SVM lineal", "svm_rbf": "SVM RBF", "random_forest": "Random Forest"}


def nombre_modelo(archivo):
    exp, clf = archivo.replace(".joblib", "").split("_", 1)
    return f"{exp} {NOMBRES_CLF.get(clf, clf)}"


def main():
    salida = DIR_RES / "galeria_errores_otros_modelos.png"
    if salida.exists():
        raise SystemExit("resultados\\fase5\\galeria_errores_otros_modelos.png ya existe; moverlo para repetir.")
    errores = [f for f in leer_csv(DIR_RES / "predicciones_prueba.csv")
               if f["acierto"] == "0" and not f["modelo"].startswith(EXCLUIR)]
    por_pieza = defaultdict(list)
    for f in errores:
        por_pieza[f["nombre_roi"]].append(f)
    if not por_pieza:
        raise SystemExit("No hay errores que mostrar.")

    filas_csv = [{"nombre_roi": f["nombre_roi"], "clase_real": f["clase_real"], "modelo": nombre_modelo(f["modelo"]),
                  "clase_predicha": f["clase_predicha"], "probabilidad": f["probabilidad_predicha"]}
                 for f in sorted(errores, key=lambda f: (f["nombre_roi"], f["modelo"]))]
    escribir_csv(DIR_RES / "galeria_errores_otros_modelos.csv", list(filas_csv[0].keys()), filas_csv)

    # Orden: primero las piezas que fallan más modelos
    piezas = sorted(por_pieza.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    cols = 3
    filas = int(np.ceil(len(piezas) / cols))
    fig, ejes = plt.subplots(filas, cols, figsize=(4.2 * cols, 5.3 * filas), squeeze=False)
    for ax in ejes.ravel():
        ax.axis("off")
    for ax, (nombre, fallos) in zip(ejes.ravel(), piezas):
        peor = max(fallos, key=lambda f: float(f["probabilidad_predicha"]))
        img = leer_png(DIR_ROI / nombre)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Real: {peor['clase_real']}\n"
                     f"Predicha: {peor['clase_predicha']} ({float(peor['probabilidad_predicha']):.2f})\n"
                     f"{nombre_modelo(peor['modelo'])}; falla en {len(fallos)} modelo{'s' if len(fallos) > 1 else ''}\n"
                     f"{nombre.replace('_p1.png', '')}",
                     fontsize=8.5, color="#b03a2e")
    fig.suptitle("Errores en prueba de los modelos distintos al final (el modelo final no falló ninguna pieza)\n"
                 "Probabilidad del modelo que se equivocó con más seguridad. Sin E3 (LBP solo).", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
    fig.savefig(salida, dpi=150)
    plt.close(fig)
    print(f"Galería con {len(piezas)} piezas; {len(errores)} errores en total.")


if __name__ == "__main__":
    main()
