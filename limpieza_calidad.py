#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filtro de calidad de las fotos base de la Parte 2 (4 clases Lego).

Qué hace: separa cada foto en pieza y fondo por color. El fondo se modela como
una superficie suave (polinomio de grado 2 por canal) ajustada con la franja del
borde de la propia foto, así una iluminación desigual no se toma por pieza.
Después mide la pieza principal y marca las fotos que pueden meter ruido al
entrenamiento. Todo se decide con la imagen: no usa nombres, fechas ni sesiones.

Banderas:
  sin_pieza         no hay ningún objeto grande en la foto
  multiples_piezas  dos o más objetos grandes (la etiqueta deja de ser clara)
  pieza_en_borde    la pieza principal toca el borde (puede estar cortada)
  borrosa           nitidez muy por debajo de la mediana de su clase.
                    Señal débil: en la prueba marca sobre todo piezas claras
                    con poco contraste, no fotos desenfocadas. Solo para mirar.
  area_atipica      área muy distinta de la mediana de su clase. Si es más del
                    doble, suele ser dos piezas pegadas en una sola mancha.
  forma_atipica     proporción muy distinta de la mediana de su clase
                    (posible etiqueta equivocada, por ejemplo un ladrillo 2x2
                    dentro de Plate_2x4). Una placa 2x4 vista en perspectiva
                    también puede salir aquí: mírela antes de excluirla.

Recomendación:
  excluir  si hay sin_pieza, multiples_piezas o pieza_en_borde
  revisar  si solo hay borrosa, area_atipica o forma_atipica
  ok       en otro caso

NO borra archivos y NO modifica exclusiones.csv. La decisión final es del
equipo, después de revisar los mosaicos.

Salidas, en documentacion\ (dentro de Taller 2):
  calidad_fotos.csv   una fila por foto, con métricas, banderas y recomendación
  revision/*.jpg      mosaicos de las fotos marcadas (pieza encuadrada)

Uso, desde la carpeta "Taller 2" con el .venv activo:
    .venv\Scripts\python limpieza_calidad.py
"""
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent          # carpeta Taller 2
DATOS = BASE / "datos_originales"
DOC = BASE / "documentacion"
REVISION = DOC / "revision"

CLASES = ["Brick_1x4", "Brick_2x2_L", "Plate_2x2", "Plate_2x4"]

AREA_MIN = 800            # px en 640x480: objetos más chicos son polvo o ruido
UMBRAL_MIN = 18           # contraste mínimo con el fondo (distancia LAB); debajo, no hay pieza
BANDA = 20                # px del borde que se usan para ajustar el fondo
RANGO_AREA = (0.5, 2.0)   # área fuera de 0.5 a 2 veces la mediana de la clase
MIN_NITIDEZ = 0.3         # nitidez menor que 0.3 veces la mediana de la clase
TOL_FORMA = 0.25          # proporción que se aleja más de 25 % de la mediana
EXCLUIR = {"sin_pieza", "multiples_piezas", "pieza_en_borde"}
BANDERAS = ["sin_pieza", "multiples_piezas", "pieza_en_borde",
            "borrosa", "area_atipica", "forma_atipica"]
COLUMNAS, CELDA, POR_MOSAICO = 4, (240, 180), 12


def leer_imagen(ruta):
    # cv2.imread falla con rutas con acentos en Windows (la carpeta tiene "José"); esto lo evita.
    return cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_COLOR)


def guardar_imagen(ruta, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    buf.tofile(str(ruta))


def superficie_fondo(lab):
    """Fondo como superficie suave (polinomio de grado 2 por canal) ajustada con
    la franja del borde. Los puntos atípicos del borde (una pieza que lo toca)
    se descartan en un segundo ajuste."""
    alto, ancho = lab.shape[:2]
    yy, xx = np.mgrid[0:alto, 0:ancho]
    x = xx / (ancho - 1) * 2 - 1
    y = yy / (alto - 1) * 2 - 1
    base = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)
    franja = np.zeros((alto, ancho), bool)
    franja[:BANDA, :] = True
    franja[-BANDA:, :] = True
    franja[:, :BANDA] = True
    franja[:, -BANDA:] = True
    A = base[franja]
    superficie = np.empty((alto, ancho, 3), np.float32)
    for c in range(3):
        b = lab[..., c][franja].astype(np.float64)
        coef = np.linalg.lstsq(A, b, rcond=None)[0]
        res = b - A @ coef
        s = 1.4826 * np.median(np.abs(res)) + 1e-6
        keep = np.abs(res) < 2.5 * s
        coef = np.linalg.lstsq(A[keep], b[keep], rcond=None)[0]
        superficie[..., c] = (base @ coef).astype(np.float32)
    return superficie


def aspecto(etiquetas, etiqueta):
    """Proporción lado largo / lado corto del rectángulo mínimo que cubre la pieza."""
    mascara = np.where(etiquetas == etiqueta, 255, 0).astype(np.uint8)
    contornos = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    contorno = max(contornos, key=cv2.contourArea)
    (_, _), (ancho, alto), _ = cv2.minAreaRect(contorno)
    return max(ancho, alto) / max(min(ancho, alto), 1.0)


def segmentar(img):
    """Devuelve la matriz de etiquetas y las piezas (objetos >= AREA_MIN), de mayor a menor."""
    alto, ancho = img.shape[:2]
    suave = cv2.medianBlur(img, 3)
    lab = cv2.cvtColor(suave, cv2.COLOR_BGR2LAB).astype(np.float32)
    fondo = superficie_fondo(lab)
    dist = np.sqrt(((lab - fondo) ** 2).sum(axis=2))
    otsu, _ = cv2.threshold(np.clip(dist, 0, 255).astype(np.uint8), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mascara = np.where(dist > max(otsu, UMBRAL_MIN), 255, 0).astype(np.uint8)
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, nucleo)
    _, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    piezas = []
    for i in range(1, len(stats)):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < AREA_MIN:
            continue
        piezas.append({
            "label": i, "x": x, "y": y, "w": w, "h": h, "area": area,
            "toca_borde": x == 0 or y == 0 or x + w >= ancho or y + h >= alto,
            "aspecto": aspecto(etiquetas, i),
        })
    piezas.sort(key=lambda p: -p["area"])
    return etiquetas, piezas


def nitidez(gris, pieza):
    """Varianza del Laplaciano en la caja de la pieza. Valores bajos = borrosa."""
    alto, ancho = gris.shape
    m = 8
    x0, y0 = max(pieza["x"] - m, 0), max(pieza["y"] - m, 0)
    x1 = min(pieza["x"] + pieza["w"] + m, ancho)
    y1 = min(pieza["y"] + pieza["h"] + m, alto)
    return float(cv2.Laplacian(gris[y0:y1, x0:x1], cv2.CV_64F).var())


def analizar(img):
    _, piezas = segmentar(img)
    r = {"n_piezas": len(piezas), "area_mayor_px": 0, "toca_borde": False,
         "nitidez": None, "aspecto": None, "piezas": piezas}
    if piezas:
        p = piezas[0]
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        r.update(area_mayor_px=p["area"], toca_borde=p["toca_borde"],
                 nitidez=round(nitidez(gris, p), 2), aspecto=round(p["aspecto"], 3))
    return r


def marcar(filas):
    """Calcula medianas por clase y asigna banderas y recomendación a cada foto."""
    por_clase = {}
    for f in filas:
        por_clase.setdefault(f["clase"], []).append(f)
    for grupo in por_clase.values():
        con_pieza = [f for f in grupo if f["n_piezas"] > 0]
        if con_pieza:
            med_area = float(np.median([f["area_mayor_px"] for f in con_pieza]))
            med_nit = float(np.median([f["nitidez"] for f in con_pieza]))
            med_asp = float(np.median([f["aspecto"] for f in con_pieza]))
        for f in grupo:
            b = []
            if f["n_piezas"] == 0:
                b.append("sin_pieza")
            if f["n_piezas"] >= 2:
                b.append("multiples_piezas")
            if f["toca_borde"]:
                b.append("pieza_en_borde")
            if f["n_piezas"] > 0:
                r = f["area_mayor_px"] / med_area
                if not RANGO_AREA[0] <= r <= RANGO_AREA[1]:
                    b.append("area_atipica")
                if f["nitidez"] < MIN_NITIDEZ * med_nit:
                    b.append("borrosa")
                if abs(f["aspecto"] / med_asp - 1) > TOL_FORMA:
                    b.append("forma_atipica")
            f["banderas"] = b
            if EXCLUIR & set(b):
                f["recomendacion"] = "excluir"
            elif b:
                f["recomendacion"] = "revisar"
            else:
                f["recomendacion"] = "ok"


def leer_exclusiones():
    ruta = DOC / "exclusiones.csv"
    if not ruta.exists():
        return set()
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        return {r["archivo"].strip() for r in csv.DictReader(f) if r.get("archivo")}


def cargar_lista():
    """Fotos base de las cuatro clases, según ImageSetKey.csv (no se escanean carpetas)."""
    lista = []
    with open(DATOS / "ImageSetKey.csv", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            clase = r["Folder2"].strip()
            if r["Folder1"] != "Base Images" or clase not in CLASES:
                continue
            nombre = r["Name"].strip()
            if not nombre.lower().endswith(".jpg"):
                nombre += ".jpg"
            lista.append((clase, nombre))
    return lista


def revisar_carpetas(lista):
    """Compara cada carpeta de clase con el CSV. Sobran = archivos .jpg que no están
    en el CSV (por ejemplo copias "(1)"). Faltan = fotos del CSV que no están en la
    carpeta. Solo informa: no borra ni mueve nada."""
    esperados = {c: set() for c in CLASES}
    for clase, nombre in lista:
        esperados[clase].add(nombre)
    print("Revisión de carpetas contra ImageSetKey.csv:")
    for clase in CLASES:
        carpeta = DATOS / "Base Images" / clase
        if not carpeta.is_dir():
            print(f"  {clase:13s} carpeta no encontrada")
            continue
        en_disco = {p.name for p in carpeta.iterdir()
                    if p.is_file() and p.suffix.lower() == ".jpg"}
        sobran = sorted(en_disco - esperados[clase])
        faltan = sorted(esperados[clase] - en_disco)
        print(f"  {clase:13s} archivos={len(en_disco):4d}  en CSV={len(esperados[clase]):4d}  "
              f"sobran={len(sobran):4d}  faltan={len(faltan):4d}")
        for nombre in sobran[:5]:
            print(f"      sobra: {nombre}")
        for nombre in faltan[:5]:
            print(f"      falta: {nombre}")


def rotular(img, texto, y, escala=0.4):
    for color, grosor in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(img, texto, (4, y), cv2.FONT_HERSHEY_SIMPLEX, escala,
                    color, grosor, cv2.LINE_AA)


def mosaico(celdas, ruta):
    """celdas: lista de (imagen, linea1, linea2). Guarda una cuadrícula."""
    ancho, alto = CELDA
    filas = (len(celdas) + COLUMNAS - 1) // COLUMNAS
    lienzo = np.full((filas * alto, COLUMNAS * ancho, 3), 255, np.uint8)
    for k, (img, l1, l2) in enumerate(celdas):
        r, c = divmod(k, COLUMNAS)
        mini = cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_AREA)
        rotular(mini, l1, 14)
        rotular(mini, l2, 30, 0.34)
        lienzo[r * alto:(r + 1) * alto, c * ancho:(c + 1) * ancho] = mini
    guardar_imagen(ruta, lienzo)


def main():
    REVISION.mkdir(parents=True, exist_ok=True)
    exclusiones = leer_exclusiones()
    filas, detalle = [], {}
    no_encontradas = no_legibles = 0
    lista = cargar_lista()
    revisar_carpetas(lista)
    for clase, nombre in lista:
        ruta = DATOS / "Base Images" / clase / nombre
        if not ruta.exists():
            no_encontradas += 1
            continue
        img = leer_imagen(ruta)
        if img is None:
            no_legibles += 1
            continue
        r = analizar(img)
        detalle[nombre] = r["piezas"]
        filas.append({"archivo": nombre, "clase": clase, "ruta": ruta,
                      "n_piezas": r["n_piezas"], "area_mayor_px": r["area_mayor_px"],
                      "toca_borde": r["toca_borde"], "nitidez": r["nitidez"],
                      "aspecto": r["aspecto"]})
    marcar(filas)

    with open(DOC / "calidad_fotos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["archivo", "clase", "n_piezas", "area_mayor_px", "toca_borde",
                    "nitidez", "aspecto", "banderas", "recomendacion", "ya_excluida"])
        for x in filas:
            w.writerow([x["archivo"], x["clase"], x["n_piezas"], x["area_mayor_px"],
                        x["toca_borde"], x["nitidez"], x["aspecto"],
                        ";".join(x["banderas"]), x["recomendacion"],
                        x["archivo"] in exclusiones])

    for clase in CLASES:
        for bandera in BANDERAS:
            grupo = [x for x in filas if x["clase"] == clase and bandera in x["banderas"]]
            for k in range(0, len(grupo), POR_MOSAICO):
                celdas = []
                for x in grupo[k:k + POR_MOSAICO]:
                    img = leer_imagen(x["ruta"])
                    for p in detalle[x["archivo"]]:
                        color = (0, 0, 255) if p["toca_borde"] else (0, 255, 0)
                        cv2.rectangle(img, (p["x"], p["y"]),
                                      (p["x"] + p["w"], p["y"] + p["h"]), color, 3)
                    celdas.append((img, x["archivo"], " ".join(x["banderas"])))
                mosaico(celdas, REVISION / f"{clase}_{bandera}_{k // POR_MOSAICO + 1:02d}.jpg")

    print(f"Fotos analizadas: {len(filas)} | no encontradas: {no_encontradas} "
          f"| no legibles: {no_legibles}")
    for clase in CLASES:
        grupo = [x for x in filas if x["clase"] == clase]
        c = Counter(x["recomendacion"] for x in grupo)
        ya = sum(x["archivo"] in exclusiones for x in grupo)
        print(f"{clase:13s} total={len(grupo):4d}  excluir={c['excluir']:4d}  "
              f"revisar={c['revisar']:4d}  ok={c['ok']:4d}  ya_excluidas={ya}")
    print(f"Tabla: {DOC / 'calidad_fotos.csv'}")
    print(f"Mosaicos para revisar: {REVISION}")


if __name__ == "__main__":
    main()
