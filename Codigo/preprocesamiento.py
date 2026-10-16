"""Preprocesamiento y segmentación de las fotos Lego (fase 2 de la Parte 2).

Qué hace: lee y normaliza una foto, la suaviza con una mediana 3x3, corrige el fondo,
segmenta las piezas y recorta el ROI de 128x128 de cada una. También trae utilidades de
lectura y escritura de archivos y de mosaicos que usan medir_fase2.py y generar_roi.py.
Cómo se corre: no se corre; se importa (medir_fase2.py, generar_roi.py y la aplicación de
la fase 6). Importarlo no lee ni escribe nada.
Qué escribe: nada por sí mismo; solo lo que otros scripts le piden con las utilidades.
"""
import csv
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
DIR_FOTOS = RAIZ / "datos_originales" / "Base Images"
DIR_PARTICIONES = RAIZ / "particiones"
CONJUNTOS = ["train", "val", "test"]
METODOS = ["a", "b0", "b1"]

LADO_MAYOR = 640
AREA_MIN_PX = 150
LADO_ROI = 128
LADO_RECORTE = 256
LADO_ESQUINA = 20
FACTOR_FONDO = 4    # el fondo se estima en una foto cuatro veces más pequeña
VENTANA_FONDO = 63  # 63 px en esa foto son 252 px en la original, más que la pieza mayor (203 px)
RADIO_MIN_PX = 9    # una mancha debe tener un punto a 9 px o más de su borde (franjas de sombra no)
NUCLEO = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# --- Banco -------------------------------------------------------------------------------

def ruta_foto(clase, archivo):
    return DIR_FOTOS / clase / archivo


def leer_particiones():
    """Fotos de train.csv, val.csv y test.csv, cada una con su conjunto. Falla si hay una
    foto duplicada o ausente en disco."""
    fotos, vistos = [], set()
    for conjunto in CONJUNTOS:
        for fila in leer_csv(DIR_PARTICIONES / f"{conjunto}.csv"):
            if fila["archivo"] in vistos:
                raise ValueError(f"foto duplicada en las particiones: {fila['archivo']}")
            vistos.add(fila["archivo"])
            if not ruta_foto(fila["clase"], fila["archivo"]).is_file():
                raise FileNotFoundError(f"falta la foto: {fila['clase']}\\{fila['archivo']}")
            fotos.append({**fila, "conjunto": conjunto})
    return fotos


# --- Pasos 1 a 3: lectura, normalización y filtro ----------------------------------------

def decodificar(ruta):
    """Paso 1: lee la foto en BGR. np.fromfile admite rutas con acento."""
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"no se pudo leer la foto: {ruta}")
    return img


def normalizar(img):
    """Paso 2: lado mayor de 640 px, con INTER_AREA."""
    alto, ancho = img.shape[:2]
    lado = max(alto, ancho)
    if lado == LADO_MAYOR:
        return img
    escala = LADO_MAYOR / lado
    return cv2.resize(img, (round(ancho * escala), round(alto * escala)), interpolation=cv2.INTER_AREA)


def leer_foto(ruta):
    return normalizar(decodificar(ruta))


def filtrar(img):
    """Paso 3: mediana 3x3 sobre BGR (Contexto, sección 1)."""
    return cv2.medianBlur(img, 3)


# --- Paso 4: corrección de fondo y pasos 5 a 7: segmentación ------------------------------

def corregir_fondo(gris):
    """Paso 4: divide cada píxel por el fondo de su zona, para que el fondo quede cerca de 255
    en toda la foto. El fondo es la mediana de una ventana mayor que cualquier pieza, así que
    las piezas casi no lo modifican. La sombra fina del borde tampoco entra en el fondo."""
    alto, ancho = gris.shape
    pequena = cv2.resize(gris, (ancho // FACTOR_FONDO, alto // FACTOR_FONDO), interpolation=cv2.INTER_AREA)
    fondo = cv2.resize(cv2.medianBlur(pequena, VENTANA_FONDO), (ancho, alto), interpolation=cv2.INTER_LINEAR)
    plano = gris.astype(np.float32) * 255.0 / np.maximum(fondo, 1).astype(np.float32)
    return np.clip(plano, 0, 255).astype(np.uint8)


def _pixeles_en_esquinas(binaria):
    n = LADO_ESQUINA
    parches = [binaria[:n, :n], binaria[:n, -n:], binaria[-n:, :n], binaria[-n:, -n:]]
    return sum(int(np.count_nonzero(p)) for p in parches)


def _umbralizar(img, metodo):
    """Paso 5. Otsu sobre el canal del método y las dos polaridades del umbral. Se queda con la
    que deja menos píxeles en las cuatro esquinas: el fondo es lo que domina las esquinas, y la
    pieza queda del lado contrario.
    a: saturación HSV (línea base). b0: gris sin corrección. b1: gris con corrección de fondo."""
    if metodo == "a":
        canal = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    elif metodo == "b0":
        canal = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif metodo == "b1":
        canal = corregir_fondo(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    else:
        raise ValueError(f"método desconocido: {metodo}")
    umbral, _ = cv2.threshold(canal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidatas = [((canal > umbral) * 255).astype(np.uint8), ((canal <= umbral) * 255).astype(np.uint8)]
    return min(candidatas, key=_pixeles_en_esquinas)


def _toca_borde(mascara):
    return bool(mascara[0, :].any() or mascara[-1, :].any() or mascara[:, 0].any() or mascara[:, -1].any())


def segmentar(img, metodo, area_min=AREA_MIN_PX):
    """Pasos 5 a 7. Devuelve las piezas de mayor a menor área; cada una es un dict con
    mascara (0/255, tamaño de la foto), area_px, caja (x, y, w, h) y toca_borde.
    El área es el número de píxeles dentro del contorno, rellenado. Las manchas que no
    llegan a 9 px de grosor (franjas de sombra del borde) no cuentan como piezas."""
    binaria = _umbralizar(img, metodo)
    # Paso 6: apertura (quita manchas) y luego cierre (rellena huecos), kernel elíptico 5x5.
    limpia = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, NUCLEO)
    limpia = cv2.morphologyEx(limpia, cv2.MORPH_CLOSE, NUCLEO)
    contornos, _ = cv2.findContours(limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    piezas = []
    for contorno in contornos:
        mascara = np.zeros(limpia.shape, np.uint8)
        cv2.drawContours(mascara, [contorno], -1, 255, thickness=cv2.FILLED)
        area = int(np.count_nonzero(mascara))
        if area < area_min:
            continue
        if cv2.distanceTransform(mascara, cv2.DIST_L2, 5).max() < RADIO_MIN_PX:
            continue
        piezas.append({"mascara": mascara, "area_px": area, "caja": cv2.boundingRect(contorno),
                       "toca_borde": _toca_borde(mascara)})
    # Desempate por posición para que el orden sea siempre el mismo.
    piezas.sort(key=lambda p: (-p["area_px"], p["caja"][1], p["caja"][0]))
    return piezas


# --- Paso 8: recorte del ROI -------------------------------------------------------------

def recorte_roi(img, mascara, caja):
    """Opción A (Contexto, sección 2): cuadrado de 256 px centrado en la caja, reducido a
    128x128 con INTER_AREA. Relleno por replicación en la imagen y con cero en la máscara."""
    x, y, w, h = caja
    mitad = LADO_RECORTE // 2
    img_p = cv2.copyMakeBorder(img, mitad, mitad, mitad, mitad, cv2.BORDER_REPLICATE)
    masc_p = cv2.copyMakeBorder(mascara, mitad, mitad, mitad, mitad, cv2.BORDER_CONSTANT, value=0)
    # En la imagen con relleno, el centro de la caja está en (x + w/2 + mitad); el recorte
    # empieza en el centro menos la mitad, es decir, en x + w/2.
    x0 = x + w // 2
    y0 = y + h // 2
    recorte_img = img_p[y0:y0 + LADO_RECORTE, x0:x0 + LADO_RECORTE]
    recorte_masc = masc_p[y0:y0 + LADO_RECORTE, x0:x0 + LADO_RECORTE]
    destino = (LADO_ROI, LADO_ROI)
    roi_img = cv2.resize(recorte_img, destino, interpolation=cv2.INTER_AREA)
    roi_masc = cv2.resize(recorte_masc, destino, interpolation=cv2.INTER_AREA)
    return roi_img, ((roi_masc > 127) * 255).astype(np.uint8)


# --- Utilidades de archivos y mosaicos -----------------------------------------------------

def leer_csv(ruta):
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def escribir_csv(ruta, columnas, filas):
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


def guardar_png(ruta, img):
    ok, datos = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"no se pudo codificar: {ruta}")
    datos.tofile(str(ruta))


def leer_png(ruta, color=True):
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)


def armar_mosaico(celdas, columnas):
    """Junta celdas del mismo tamaño en una cuadrícula; las que faltan quedan en negro."""
    alto, ancho = celdas[0].shape[:2]
    filas = -(-len(celdas) // columnas)
    lienzo = np.zeros((filas * alto, columnas * ancho, 3), np.uint8)
    for i, celda in enumerate(celdas):
        f, c = divmod(i, columnas)
        lienzo[f * alto:(f + 1) * alto, c * ancho:(c + 1) * ancho] = celda
    return lienzo


def rotular(celda, texto):
    """Escribe un texto pequeño en la esquina de la celda, con sombra para leerlo sobre cualquier fondo."""
    cv2.putText(celda, texto, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(celda, texto, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return celda
