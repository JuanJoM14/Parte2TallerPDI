"""Genera el banco de ROI de la fase 2: un recorte de 128x128 y su máscara por pieza que entra a
un conjunto.

Qué hace: aplica el método y el filtro de resultados\\fase2\\parametros.json a las fotos del banco
(mediana 3x3 solo si el filtro dice "mediana_3"), recorta cada pieza (opción A) y asigna el
conjunto de uso según Fase2_Contexto.md, sección 4. Las fotos de piezas dobles no entran al banco
(ver fotos_con_piezas_dobles). Las piezas con uso "no_usada" no se guardan.
Cómo se corre (desde Taller 2):  .venv\\Scripts\\python Codigo\\generar_roi.py
Qué escribe: roi\\imagenes\\ y roi\\mascaras\\ (un PNG de 128x128 por pieza usada; la máscara solo tiene
0 y 255), roi\\metadatos.csv (una fila por pieza usada); en resultados\\fase2\\: resumen_roi.csv (todas
las piezas segmentadas), piezas_no_usadas.csv (las que no se guardan, con su motivo),
piezas_mayores_256.csv (todas las piezas segmentadas) y fotos_excluidas.csv; en
resultados\\fase2\\mosaicos\\: roi_control.png.
Si roi\\metadatos.csv ya existe, se detiene. Para regenerar, mover roi\\ fuera de Taller 2 antes.
"""
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from preprocesamiento import (CONJUNTOS, LADO_RECORTE, METODOS, RAIZ, armar_mosaico, escribir_csv, filtrar,
                              guardar_png, leer_csv, leer_foto, leer_particiones, leer_png, recorte_roi,
                              ruta_foto, rotular, segmentar)

DIR_ROI = RAIZ / "roi"
DIR_IMG = DIR_ROI / "imagenes"
DIR_MASC = DIR_ROI / "mascaras"
DIR_RES = RAIZ / "resultados" / "fase2"
DIR_MOSAICOS = DIR_RES / "mosaicos"
SEMILLA = 42
FOTOS_ESPERADAS = 974  # fotos base; el banco usa estas menos las de piezas dobles
CELDAS_CONTROL = 24
COLUMNAS_CONTROL = 6
FILTROS = ("mediana_3", "ninguno")
COLUMNAS = ["archivo", "clase", "conjunto", "uso", "bloque", "n_piezas", "indice_pieza", "es_central",
            "etiquetada", "toca_borde", "caja_x", "caja_y", "caja_ancho", "caja_alto", "area_px",
            "supera_256", "nombre_roi"]
COLUMNAS_NO_USADAS = ["archivo", "clase", "conjunto", "n_piezas", "indice_pieza", "nombre_roi", "motivo"]


# --- Reglas de conjunto (Fase2_Contexto.md, sección 4) ---------------------------------------

def filas_de_foto(foto, img, piezas):
    """Una fila por pieza. Es central la pieza más cercana al centro de la foto. Con dos o más
    piezas, en entrenamiento no entra ninguna; en validación y prueba, solo la central."""
    alto, ancho = img.shape[:2]
    distancias = [np.hypot(p["caja"][0] + p["caja"][2] / 2 - ancho / 2, p["caja"][1] + p["caja"][3] / 2 - alto / 2)
                  for p in piezas]
    central = int(np.argmin(distancias))
    n = len(piezas)
    filas = []
    for k, p in enumerate(piezas, start=1):
        x, y, w, h = p["caja"]
        es_central = int(k - 1 == central)
        etiquetada = int(n == 1 or (foto["conjunto"] != "train" and es_central))
        if foto["conjunto"] == "train":
            uso = "train" if (etiquetada and not p["toca_borde"]) else "no_usada"
        else:
            uso = foto["conjunto"] if etiquetada else "no_usada"
        filas.append({"archivo": foto["archivo"], "clase": foto["clase"], "conjunto": foto["conjunto"],
                      "uso": uso, "bloque": foto["bloque"], "n_piezas": n, "indice_pieza": k,
                      "es_central": es_central, "etiquetada": etiquetada, "toca_borde": int(p["toca_borde"]),
                      "caja_x": x, "caja_y": y, "caja_ancho": w, "caja_alto": h, "area_px": p["area_px"],
                      "supera_256": int(w > LADO_RECORTE or h > LADO_RECORTE),
                      "nombre_roi": f"{Path(foto['archivo']).stem}_p{k}.png"})
    return filas


def motivo_no_usada(fila):
    """Por qué una pieza no entra a ningún conjunto (para piezas_no_usadas.csv)."""
    motivos = []
    if fila["n_piezas"] > 1 and fila["conjunto"] == "train":
        motivos.append("foto de dos piezas: entrenamiento no usa ninguna")
    if fila["n_piezas"] > 1 and fila["conjunto"] != "train":
        motivos.append("no es la central de una foto de dos piezas")
    if fila["toca_borde"] and fila["conjunto"] == "train":
        motivos.append("toca el borde")
    return "; ".join(motivos)


def escribir_recortes(img, piezas, filas):
    """PNG de imagen y de máscara por pieza usada, con el mismo nombre."""
    for p, fila in zip(piezas, filas):
        if fila["uso"] == "no_usada":
            continue
        roi_img, roi_masc = recorte_roi(img, p["mascara"], p["caja"])
        guardar_png(DIR_IMG / fila["nombre_roi"], roi_img)
        guardar_png(DIR_MASC / fila["nombre_roi"], roi_masc)


# --- Resúmenes y control ----------------------------------------------------------------------

def fotos_con_piezas_dobles():
    """Fotos que la revisión a mano de contornos_grandes.csv marcó como dos piezas que se tocan
    (revision_manual empieza con "dos piezas"). Salen de los tres conjuntos (Fase2_Contexto, sección 4)."""
    ruta = DIR_RES / "contornos_grandes.csv"
    if not ruta.exists():
        raise SystemExit("Falta resultados\\fase2\\contornos_grandes.csv. Correr medir_fase2.py antes.")
    dobles = {}
    for fila in leer_csv(ruta):
        revision = (fila.get("revision_manual") or "").strip()
        if revision.lower().startswith("dos piezas"):
            dobles[fila["archivo"]] = revision
    return dobles


def resumen(fotos, filas):
    """Conteos por conjunto para la sección 'Verificación al cerrar' del plan."""
    piezas_por_foto = Counter(r["archivo"] for r in filas)
    tabla = []
    for conjunto in CONJUNTOS:
        de_conjunto = [r for r in filas if r["conjunto"] == conjunto]
        fotos_c = [f["archivo"] for f in fotos if f["conjunto"] == conjunto]
        una = sum(1 for a in fotos_c if piezas_por_foto[a] == 1)
        tabla.append({"conjunto": conjunto, "fotos": len(fotos_c), "fotos_1_pieza": una,
                      "fotos_2_o_mas": len(fotos_c) - una, "piezas": len(de_conjunto),
                      "piezas_en_uso": sum(1 for r in de_conjunto if r["uso"] == conjunto),
                      "piezas_que_tocan_borde": sum(r["toca_borde"] for r in de_conjunto),
                      "piezas_supera_256": sum(r["supera_256"] for r in de_conjunto)})
    return tabla


def mosaico_control(filas):
    """24 ROI al azar (semilla 42) con su contorno en verde."""
    elegidas = random.Random(SEMILLA).sample(filas, min(CELDAS_CONTROL, len(filas)))
    celdas = []
    for n, fila in enumerate(elegidas, start=1):
        roi = leer_png(DIR_IMG / fila["nombre_roi"])
        masc = leer_png(DIR_MASC / fila["nombre_roi"], color=False)
        contornos, _ = cv2.findContours(masc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, contornos, -1, (0, 255, 0), 1)
        celdas.append(rotular(roi, f"{n} {fila['uso']}"))
    guardar_png(DIR_MOSAICOS / "roi_control.png", armar_mosaico(celdas, COLUMNAS_CONTROL))


# --- Programa principal ---------------------------------------------------------------------------

def cargar_parametros():
    ruta = DIR_RES / "parametros.json"
    if not ruta.exists():
        raise SystemExit("Falta resultados\\fase2\\parametros.json. Correr medir_fase2.py antes.")
    params = json.loads(ruta.read_text(encoding="utf-8"))
    if params.get("segmentacion") not in METODOS:
        raise SystemExit(f"método desconocido en parametros.json: {params.get('segmentacion')}")
    if params.get("filtro") not in FILTROS:
        raise SystemExit(f"filtro desconocido en parametros.json: {params.get('filtro')}")
    return params


def main():
    if (DIR_ROI / "metadatos.csv").exists():
        raise SystemExit("roi\\metadatos.csv ya existe. Para regenerar, mover roi\\ fuera de Taller 2 antes.")
    params = cargar_parametros()
    metodo = params["segmentacion"]
    fotos = leer_particiones()
    assert len(fotos) == FOTOS_ESPERADAS, f"se esperaban {FOTOS_ESPERADAS} fotos y hay {len(fotos)}"
    dobles = fotos_con_piezas_dobles()
    excluidas = [f for f in fotos if f["archivo"] in dobles]
    assert len(excluidas) == len(dobles), "hay fotos marcadas como piezas dobles que no están en las particiones"
    fotos = [f for f in fotos if f["archivo"] not in dobles]
    for carpeta in (DIR_IMG, DIR_MASC, DIR_MOSAICOS):
        carpeta.mkdir(parents=True, exist_ok=True)

    filas, sin_pieza = [], []
    for foto in fotos:
        img = leer_foto(ruta_foto(foto["clase"], foto["archivo"]))
        if params["filtro"] == "mediana_3":
            img = filtrar(img)
        piezas = segmentar(img, metodo)
        if not piezas:
            sin_pieza.append(foto["archivo"])
            continue
        filas_foto = filas_de_foto(foto, img, piezas)
        escribir_recortes(img, piezas, filas_foto)
        filas += filas_foto
    if sin_pieza:
        raise SystemExit(f"fotos sin ninguna pieza: {sin_pieza}. No se escribió metadatos.csv.")

    filas.sort(key=lambda r: (r["archivo"], r["indice_pieza"]))
    tabla = resumen(fotos, filas)
    usadas = [r for r in filas if r["uso"] != "no_usada"]
    no_usadas = [r for r in filas if r["uso"] == "no_usada"]
    escribir_csv(DIR_ROI / "metadatos.csv", COLUMNAS, usadas)
    escribir_csv(DIR_RES / "resumen_roi.csv", list(tabla[0].keys()), tabla)
    escribir_csv(DIR_RES / "fotos_excluidas.csv", ["archivo", "clase", "conjunto", "motivo"],
                 [{"archivo": f["archivo"], "clase": f["clase"], "conjunto": f["conjunto"],
                   "motivo": dobles[f["archivo"]]} for f in excluidas])
    escribir_csv(DIR_RES / "piezas_no_usadas.csv", COLUMNAS_NO_USADAS,
                 [{c: (motivo_no_usada(r) if c == "motivo" else r[c]) for c in COLUMNAS_NO_USADAS}
                  for r in no_usadas])
    escribir_csv(DIR_RES / "piezas_mayores_256.csv", COLUMNAS, [r for r in filas if r["supera_256"]])
    mosaico_control(usadas)

    print(f"Método: {metodo}. Filtro: {params['filtro']}. Fotos: {len(fotos)}. Piezas segmentadas: {len(filas)}. "
          f"Guardadas: {len(usadas)}. No guardadas: {len(no_usadas)} (ver piezas_no_usadas.csv).")
    por_conjunto = {c: sum(1 for f in excluidas if f["conjunto"] == c) for c in CONJUNTOS}
    print(f"Excluidas por piezas dobles: {len(excluidas)} (train {por_conjunto['train']}, "
          f"val {por_conjunto['val']}, test {por_conjunto['test']}). Ver fotos_excluidas.csv.")
    for fila in tabla:
        print("  " + "; ".join(f"{k} {v}" for k, v in fila.items()))


if __name__ == "__main__":
    main()
