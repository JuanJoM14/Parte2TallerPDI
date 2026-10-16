"""Genera las 10 escenas de varias piezas y sus variantes de luz y fondo (fase 6).

Qué hace (detalle en documentacion\\PlanesContextoDeFase\\Fase6_Contexto.md, filas 6 a 9):
  - Toma piezas de fotos de PRUEBA de una sola pieza (roi\\metadatos.csv), sin repetir ninguna.
  - Cada pieza se recorta de su foto original (640 px) con la máscara de la segmentación de la
    fase 2 y se pega a tamaño real, sin girar ni escalar, sobre
    datos_originales\\background_backlit_A.jpg, sin solaparse y a 10 px o más de otras piezas.
  - De 3 a 5 piezas por escena, semilla 42.
  - Cuatro variantes por escena: oscura (x0,6), clara (x1,4), luz desigual (gradiente de x0,5 a
    x1,2 de izquierda a derecha) y otro fondo (gris con textura, generado).
Cómo se corre (desde Taller 2):  .venv\\Scripts\\python Codigo\\generar_escenas.py
Qué escribe: escenas\\ (50 PNG) y escenas\\verdad.csv (una fila por pieza pegada y por imagen).
Se detiene si escenas\\verdad.csv ya existe.
"""
import random

import cv2
import numpy as np

from preprocesamiento import (RAIZ, escribir_csv, filtrar, guardar_png, leer_csv, leer_foto, ruta_foto,
                              segmentar)

DIR_ESC = RAIZ / "escenas"
RUTA_FONDO = RAIZ / "datos_originales" / "background_backlit_A.jpg"
SEMILLA = 42
N_ESCENAS = 10
PIEZAS_MIN, PIEZAS_MAX = 3, 5
SEPARACION = 10
INTENTOS = 500
VARIANTES = ("oscura", "clara", "luz_desigual", "otro_fondo")


def elegir_piezas(rng):
    """Piezas de prueba de fotos con una sola pieza; cada escena con 3 a 5, sin repetir."""
    filas = [f for f in leer_csv(RAIZ / "roi" / "metadatos.csv")
             if f["uso"] == "test" and f["n_piezas"] == "1"]
    filas.sort(key=lambda f: f["nombre_roi"])
    rng.shuffle(filas)
    escenas, i = [], 0
    for _ in range(N_ESCENAS):
        k = rng.randint(PIEZAS_MIN, PIEZAS_MAX)
        escenas.append(filas[i:i + k])
        i += k
    return escenas


def recortar_pieza(fila):
    """Pieza de la foto original de 640 px con su máscara (la mayor, como en la fase 2)."""
    img = leer_foto(ruta_foto(fila["clase"], fila["archivo"]))
    pieza = segmentar(filtrar(img), "b1")[0]
    x, y, w, h = pieza["caja"]
    return img[y:y + h, x:x + w].copy(), pieza["mascara"][y:y + h, x:x + w].copy()


def ubicar(rng, ocupado, w, h, alto, ancho):
    for _ in range(INTENTOS):
        x = rng.randint(SEPARACION, ancho - w - SEPARACION)
        y = rng.randint(SEPARACION, alto - h - SEPARACION)
        zona = ocupado[max(0, y - SEPARACION):y + h + SEPARACION, max(0, x - SEPARACION):x + w + SEPARACION]
        if not zona.any():
            return x, y
    return None


def fondo_textura(alto, ancho, semilla):
    """Fondo gris con textura suave, distinto del montaje del banco."""
    rs = np.random.RandomState(semilla)
    ruido = cv2.GaussianBlur(rs.normal(0, 18, (alto, ancho)).astype(np.float32), (0, 0), 3)
    base = 150 + ruido
    return np.clip(np.dstack([base, base * 0.97, base * 0.92]), 0, 255).astype(np.uint8)


def ajustar(img, factor):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def main():
    if (DIR_ESC / "verdad.csv").exists():
        raise SystemExit("escenas\\verdad.csv ya existe; mover escenas\\ fuera de Taller 2 para regenerar.")
    rng = random.Random(SEMILLA)
    fondo = leer_foto(RUTA_FONDO)
    alto, ancho = fondo.shape[:2]
    textura = fondo_textura(alto, ancho, SEMILLA)
    gradiente = np.linspace(0.5, 1.2, ancho, dtype=np.float32)[None, :, None]
    DIR_ESC.mkdir(exist_ok=True)
    verdad = []
    for n, grupo in enumerate(elegir_piezas(rng), start=1):
        escena, otro = fondo.copy(), textura.copy()
        ocupado = np.zeros((alto, ancho), bool)
        pegadas = []
        for fila in grupo:
            recorte, masc = recortar_pieza(fila)
            h, w = masc.shape
            pos = ubicar(rng, ocupado, w, h, alto, ancho)
            if pos is None:
                print(f"escena {n:02d}: no cupo {fila['nombre_roi']} ({w}x{h}); se omite")
                continue
            x, y = pos
            sel = masc > 0
            escena[y:y + h, x:x + w][sel] = recorte[sel]
            otro[y:y + h, x:x + w][sel] = recorte[sel]
            ocupado[y:y + h, x:x + w] = True
            pegadas.append({"clase": fila["clase"], "origen": fila["archivo"], "x": x, "y": y, "w": w, "h": h})
        imagenes = {"original": escena, "oscura": ajustar(escena, 0.6), "clara": ajustar(escena, 1.4),
                    "luz_desigual": np.clip(escena.astype(np.float32) * gradiente, 0, 255).astype(np.uint8),
                    "otro_fondo": otro}
        for tipo, img in imagenes.items():
            nombre = f"escena_{n:02d}.png" if tipo == "original" else f"escena_{n:02d}_{tipo}.png"
            guardar_png(DIR_ESC / nombre, img)
            for k, p in enumerate(pegadas, start=1):
                verdad.append({"imagen": nombre, "escena": n, "tipo": tipo, "pieza": k, **p})
        print(f"escena {n:02d}: {len(pegadas)} piezas")
    escribir_csv(DIR_ESC / "verdad.csv", list(verdad[0].keys()), verdad)
    print(f"Listo: {N_ESCENAS} escenas, {N_ESCENAS * len(VARIANTES)} variantes, "
          f"{sum(1 for v in verdad if v['tipo'] == 'original')} piezas.")


if __name__ == "__main__":
    main()
