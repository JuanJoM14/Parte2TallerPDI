"""Extrae las características de la fase 3 sobre el banco de ROI (roi\\).

Qué hace: por cada fila de roi\\metadatos.csv lee el recorte 128x128 y su máscara y calcula
  - HOG: 9 orientaciones, celdas 8x8, bloques 2x2 (L2-Hys) sobre el gris del recorte.
  - Forma (sobre el contorno mayor de la máscara): área, perímetro, proporción (lado mayor /
    lado menor del rectángulo mínimo), solidez y los 7 momentos de Hu en escala log
    (-signo * log10|hu|).
  - Textura LBP: patrones binarios locales uniformes invariantes a rotación, en dos escalas
    (P=8, R=1 y P=16, R=2, como pide el enunciado) sobre el gris del recorte; un histograma
    normalizado por escala (10 y 18 bins), solo con los píxeles de la máscara.
Área y perímetro están en píxeles del recorte de 128 (escala 1/2 de la foto de 640).
No escala ni aplica PCA: eso se ajusta en la fase de modelos, solo con entrenamiento.
Cómo se corre (desde Taller 2):  .venv\\Scripts\\python Codigo\\caracteristicas.py
Qué escribe: caracteristicas\\{train,val,test}.npz (X, y, nombres, columnas) y
caracteristicas\\resumen.json (dimensiones por bloque y conteos). Se detiene si ya existen.
"""
import json

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from preprocesamiento import RAIZ, leer_csv, leer_png

DIR_ROI = RAIZ / "roi"
DIR_SAL = RAIZ / "caracteristicas"
USOS = ("train", "val", "test")
FILAS_ESPERADAS = 951
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                  block_norm="L2-Hys", feature_vector=True)
COLS_FORMA = ["area", "perimetro", "proporcion", "solidez"] + [f"hu{i}" for i in range(1, 8)]
LBP_ESCALAS = ((8, 1), (16, 2))  # (P, R)
# método "uniform": P+1 patrones uniformes y 1 bin para los no uniformes -> P+2 bins por escala
COLS_LBP = [f"lbp{p}r{r}_{i}" for p, r in LBP_ESCALAS for i in range(p + 2)]


def car_hog(img_bgr):
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return hog(gris, **HOG_PARAMS).astype(np.float32)


def car_forma(masc):
    contornos, _ = cv2.findContours((masc > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    if not contornos:
        raise ValueError("máscara vacía")
    c = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimetro = cv2.arcLength(c, True)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    proporcion = max(w, h) / max(min(w, h), 1e-6)
    area_casco = cv2.contourArea(cv2.convexHull(c))
    solidez = area / area_casco if area_casco > 0 else 0.0
    hu = cv2.HuMoments(cv2.moments(c)).ravel()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return np.array([area, perimetro, proporcion, solidez, *hu_log], dtype=np.float32)


def car_lbp(img_bgr, masc):
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    partes = []
    for p, r in LBP_ESCALAS:
        codigos = local_binary_pattern(gris, p, r, method="uniform")
        hist = np.bincount(codigos[masc > 0].astype(np.int64), minlength=p + 2).astype(np.float64)
        partes.append(hist / max(hist.sum(), 1.0))
    return np.concatenate(partes).astype(np.float32)


def extraer(fila):
    img = leer_png(DIR_ROI / "imagenes" / fila["nombre_roi"])
    masc = leer_png(DIR_ROI / "mascaras" / fila["nombre_roi"], color=False)
    return np.concatenate([car_hog(img), car_forma(masc), car_lbp(img, masc)])


def main():
    if (DIR_SAL / "resumen.json").exists():
        raise SystemExit("caracteristicas\\resumen.json ya existe; mover la carpeta para regenerar.")
    filas = leer_csv(DIR_ROI / "metadatos.csv")
    if len(filas) != FILAS_ESPERADAS:
        raise SystemExit(f"Se esperaban {FILAS_ESPERADAS} filas y hay {len(filas)}.")
    dim_hog = len(car_hog(np.zeros((128, 128, 3), np.uint8)))
    columnas = [f"hog{i}" for i in range(dim_hog)] + COLS_FORMA + COLS_LBP
    DIR_SAL.mkdir(exist_ok=True)
    resumen = {"dim_hog": dim_hog, "dim_forma": len(COLS_FORMA), "dim_lbp": len(COLS_LBP),
               "dim_total": len(columnas), "hog": {k: str(v) for k, v in HOG_PARAMS.items()},
               "lbp": "uniform; escalas (P, R) = " + str(list(LBP_ESCALAS))
                      + "; un histograma normalizado por escala, dentro de la máscara",
               "conteos": {}}
    for uso in USOS:
        sub = [f for f in filas if f["uso"] == uso]
        X = np.stack([extraer(f) for f in sub])
        if not np.isfinite(X).all():
            raise SystemExit(f"Valores no finitos en {uso}.")
        np.savez_compressed(DIR_SAL / f"{uso}.npz", X=X, y=np.array([f["clase"] for f in sub]),
                            nombres=np.array([f["nombre_roi"] for f in sub]),
                            columnas=np.array(columnas))
        resumen["conteos"][uso] = len(sub)
        print(f"{uso}: {X.shape}")
    (DIR_SAL / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(f"Dimensión HOG: {dim_hog}; total: {len(columnas)}")


if __name__ == "__main__":
    main()
