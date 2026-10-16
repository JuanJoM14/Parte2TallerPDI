"""Mediciones de la fase 2 que fijan el método de segmentación (versión 2: corrección de fondo).

Qué hace, en dos pasos:
1. Muestra 2: 60 fotos de entrenamiento que no están en la muestra 1, 15 por clase, elegidas
   con random.Random(42). Escribe la lista y tres hojas para contar las piezas a mano.
2. Con el conteo a mano (resultados\\fase2\\muestra2_conteo_a_mano.csv) compara los métodos
   (a), (b0) y (b1), elige el ganador por la regla del plan y, si el ganador llega a 54 de 60,
   mide el filtro en el entrenamiento y lista los contornos grandes de las 974 fotos.
Cómo se corre (desde Taller 2):  .venv\\Scripts\\python Codigo\\medir_fase2.py
Qué escribe, en resultados\\fase2\\: muestra2_seleccion.csv, medicion_luz.csv (diagnóstico),
medicion_muestra1.csv (diagnóstico), medicion_segmentacion.csv, parametros.json,
medicion_filtro.csv, contornos_grandes.csv; y en mosaicos\\: muestra2_hoja_N.png,
control_<método>.png, contornos_grandes.png.
Si falta muestra2_conteo_a_mano.csv, solo hace el paso 1 y se detiene.
Si contornos_grandes.csv tiene revisiones a mano escritas, se detiene antes de tocar nada.
"""
import json
import random
import statistics
from collections import defaultdict

import cv2
import numpy as np

from preprocesamiento import (AREA_MIN_PX, FACTOR_FONDO, RADIO_MIN_PX, RAIZ, VENTANA_FONDO, armar_mosaico,
                              escribir_csv, filtrar, guardar_png, leer_csv, leer_foto, leer_particiones,
                              recorte_roi, ruta_foto, rotular, segmentar)

DIR_RES = RAIZ / "resultados" / "fase2"
DIR_MOSAICOS = DIR_RES / "mosaicos"
METODOS = ["a", "b0", "b1"]
CLASES = ["Brick_1x4", "Brick_2x2_L", "Plate_2x2", "Plate_2x4"]
SEMILLA = 42
POR_CLASE = 15
MINIMO_ACIERTOS = 54  # de 60: si el ganador no llega, se detiene antes del filtro y los contornos
FRANJA_BORDE_PX = 10
FACTOR_GRANDE = 1.5
COLUMNAS_HOJA = 5
FILAS_HOJA = 4
COLUMNAS_CONTROL = 6
CELDAS_CONTROL = 24


# --- Medición 0: luz (diagnóstico, no decide nada) ------------------------------------------

def brillo_borde(img):
    """Mediana del canal L de Lab en la franja de 10 px del borde."""
    l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
    interior = np.zeros(l.shape, bool)
    interior[FRANJA_BORDE_PX:-FRANJA_BORDE_PX, FRANJA_BORDE_PX:-FRANJA_BORDE_PX] = True
    return float(np.median(l[~interior]))


def medir_luz(fotos):
    """Variación del brillo del borde entre fotos. Se guarda para documentar por qué la regla
    entre fotos no basta: no ve el gradiente dentro de cada foto."""
    entrenamiento = [f for f in fotos if f["conjunto"] == "train"]
    filas = [{"archivo": f["archivo"], "clase": f["clase"], "bloque": f["bloque"],
              "brillo_borde": round(brillo_borde(leer_foto(ruta_foto(f["clase"], f["archivo"]))), 1)}
             for f in entrenamiento]
    escribir_csv(DIR_RES / "medicion_luz.csv", ["archivo", "clase", "bloque", "brillo_borde"], filas)
    p5, p95 = np.percentile([f["brillo_borde"] for f in filas], [5, 95])
    print(f"0. Luz (diagnóstico): entre fotos, percentil 5 = {p5:.1f}, percentil 95 = {p95:.1f}, "
          f"dispersión = {p95 - p5:.1f}. No decide nada.")


# --- Paso 1: muestra 2 -------------------------------------------------------------------------

def seleccionar_muestra2(fotos):
    """60 fotos de entrenamiento que no están en la muestra 1: 15 por clase, con random.Random(42)
    sobre la lista ordenada por nombre de archivo. Devuelve las fotos en orden de clase y nombre."""
    usadas = {fila["archivo"] for fila in leer_csv(DIR_RES / "muestra_medicion.csv")}
    azar = random.Random(SEMILLA)
    elegidas = []
    for clase in CLASES:
        candidatas = sorted((f for f in fotos if f["conjunto"] == "train" and f["clase"] == clase
                             and f["archivo"] not in usadas), key=lambda f: f["archivo"])
        elegidas += azar.sample(candidatas, POR_CLASE)
    elegidas.sort(key=lambda f: (CLASES.index(f["clase"]), f["archivo"]))
    return elegidas


def hojas_para_contar(elegidas):
    """Fotos sin contornos, con su número, para contar las piezas a mano."""
    celdas = []
    for n, foto in enumerate(elegidas, start=1):
        img = cv2.resize(leer_foto(ruta_foto(foto["clase"], foto["archivo"])), (320, 240),
                         interpolation=cv2.INTER_AREA)
        celdas.append(rotular(img, str(n)))
    por_hoja = COLUMNAS_HOJA * FILAS_HOJA
    for h in range(0, len(celdas), por_hoja):
        hoja = armar_mosaico(celdas[h:h + por_hoja], COLUMNAS_HOJA)
        guardar_png(DIR_MOSAICOS / f"muestra2_hoja_{h // por_hoja + 1}.png", hoja)


# --- Paso 2: comparación de métodos sobre una muestra con conteo a mano --------------------------

def analizar_muestra(filas_mano, por_nombre):
    """Para cada foto de la muestra: la imagen filtrada, el conteo a mano y las piezas de cada método."""
    muestra = []
    for fila in filas_mano:
        foto = por_nombre[fila["archivo"]]
        img = filtrar(leer_foto(ruta_foto(foto["clase"], foto["archivo"])))
        muestra.append({"indice": fila["indice"], "archivo": fila["archivo"], "clase": fila["clase"],
                        "a_mano": int(fila["n_piezas_a_mano"]), "img": img,
                        "piezas": {m: segmentar(img, m) for m in METODOS}})
    return muestra


def tabla_coincidencias(muestra):
    filas = []
    for s in muestra:
        fila = {"indice": s["indice"], "archivo": s["archivo"], "clase": s["clase"], "n_piezas_a_mano": s["a_mano"]}
        for m in METODOS:
            fila[f"n_{m}"] = len(s["piezas"][m])
            fila[f"coincide_{m}"] = int(len(s["piezas"][m]) == s["a_mano"])
        filas.append(fila)
    return filas


def celda_con_contorno(img, pieza, etiqueta):
    """ROI de la pieza con su contorno en verde y una etiqueta."""
    roi, masc = recorte_roi(img, pieza["mascara"], pieza["caja"])
    contornos, _ = cv2.findContours(masc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(roi, contornos, -1, (0, 255, 0), 1)
    return rotular(roi, etiqueta)


def mosaico_control(muestra, metodo):
    """24 ROI elegidos al azar (semilla 42) entre las piezas que encontró el método."""
    candidatas = [(s, k, p) for s in muestra for k, p in enumerate(s["piezas"][metodo], start=1)]
    elegidas = random.Random(SEMILLA).sample(candidatas, min(CELDAS_CONTROL, len(candidatas)))
    celdas = [celda_con_contorno(s["img"], p, f"{s['indice']}-p{k}") for s, k, p in elegidas]
    guardar_png(DIR_MOSAICOS / f"control_{metodo}.png", armar_mosaico(celdas, COLUMNAS_CONTROL))


def elegir_metodo(tabla):
    """Gana el método con más coincidencias; en empate, (a). Empate sin (a): se detiene."""
    aciertos = {m: sum(f[f"coincide_{m}"] for f in tabla) for m in METODOS}
    mejor = max(aciertos.values())
    empatados = [m for m in METODOS if aciertos[m] == mejor]
    if len(empatados) > 1 and "a" not in empatados:
        raise SystemExit(f"Empate entre {empatados} sin (a): la regla del plan no lo resuelve. "
                         f"Coincidencias: {aciertos}. No se escribió parametros.json.")
    return ("a" if "a" in empatados else empatados[0]), aciertos


# --- Paso 3: filtro (con el método elegido, en entrenamiento) -------------------------------------

def contar(fotos, metodo, con_filtro):
    """Piezas de cada foto sin la máscara (área, caja y borde); la lista completa de piezas
    se pide solo cuando hace falta."""
    resultado = {}
    for f in fotos:
        img = leer_foto(ruta_foto(f["clase"], f["archivo"]))
        if con_filtro:
            img = filtrar(img)
        resultado[f["archivo"]] = [{k: p[k] for k in ("area_px", "caja", "toca_borde")}
                                   for p in segmentar(img, metodo)]
    return resultado


def medir_filtro(fotos, metodo):
    entrenamiento = [f for f in fotos if f["conjunto"] == "train"]
    sin = contar(entrenamiento, metodo, con_filtro=False)
    con = contar(entrenamiento, metodo, con_filtro=True)
    filas = [{"archivo": f["archivo"], "clase": f["clase"], "n_sin_filtro": len(sin[f["archivo"]]),
              "n_con_mediana": len(con[f["archivo"]]),
              "cambia": int(len(sin[f["archivo"]]) != len(con[f["archivo"]]))} for f in entrenamiento]
    escribir_csv(DIR_RES / "medicion_filtro.csv", ["archivo", "clase", "n_sin_filtro", "n_con_mediana", "cambia"], filas)
    cambian = sum(f["cambia"] for f in filas)
    print(f"3. Filtro: {cambian} de {len(filas)} fotos de entrenamiento cambian de número de piezas con la mediana")


# --- Paso 4: contornos grandes (piezas que se tocan) ----------------------------------------------

def medianas_por_clase(fotos, piezas_por_foto):
    """Mediana de área por clase con las fotos de entrenamiento que tienen una sola pieza."""
    areas = defaultdict(list)
    for f in fotos:
        piezas = piezas_por_foto[f["archivo"]]
        if f["conjunto"] == "train" and len(piezas) == 1:
            areas[f["clase"]].append(piezas[0]["area_px"])
    return {c: statistics.median(areas[c]) for c in CLASES}


def listar_contornos_grandes(fotos, piezas_por_foto, medianas):
    """Contornos de las 974 fotos con área mayor que 1,5 veces la mediana de su clase."""
    filas = []
    for f in fotos:
        limite = FACTOR_GRANDE * medianas[f["clase"]]
        for p in piezas_por_foto[f["archivo"]]:
            if p["area_px"] > limite:
                filas.append({"archivo": f["archivo"], "clase": f["clase"], "conjunto": f["conjunto"],
                              "area_px": p["area_px"], "mediana_clase_px": medianas[f["clase"]],
                              "revision_manual": ""})
    filas.sort(key=lambda r: (CLASES.index(r["clase"]), r["archivo"], -r["area_px"]))
    return filas


def mosaico_contornos_grandes(filas, metodo, por_nombre):
    """Un ROI con contorno por fila de contornos_grandes.csv; la etiqueta es el número de fila."""
    celdas = []
    for n, fila in enumerate(filas, start=1):
        foto = por_nombre[fila["archivo"]]
        img = filtrar(leer_foto(ruta_foto(foto["clase"], foto["archivo"])))
        pieza = next(p for p in segmentar(img, metodo) if p["area_px"] == fila["area_px"])
        celdas.append(celda_con_contorno(img, pieza, str(n)))
    guardar_png(DIR_MOSAICOS / "contornos_grandes.png", armar_mosaico(celdas, COLUMNAS_CONTROL))


def medir_contornos_grandes(fotos, metodo):
    piezas = contar(fotos, metodo, con_filtro=True)
    medianas = medianas_por_clase(fotos, piezas)
    filas = listar_contornos_grandes(fotos, piezas, medianas)
    escribir_csv(DIR_RES / "contornos_grandes.csv",
                 ["archivo", "clase", "conjunto", "area_px", "mediana_clase_px", "revision_manual"], filas)
    if filas:
        mosaico_contornos_grandes(filas, metodo, {f["archivo"]: f for f in fotos})
    por_conjunto = {c: sum(1 for r in filas if r["conjunto"] == c) for c in ("train", "val", "test")}
    print("4. Contornos grandes (> 1,5 veces la mediana de la clase):")
    print("   medianas de área: " + ", ".join(f"{c} {medianas[c]:g}" for c in CLASES))
    print(f"   {len(filas)} contornos: {por_conjunto}")


# --- Programa principal -----------------------------------------------------------------------------

def revisiones_escritas():
    ruta = DIR_RES / "contornos_grandes.csv"
    return ruta.exists() and any((f.get("revision_manual") or "").strip() for f in leer_csv(ruta))


def main():
    if revisiones_escritas():
        raise SystemExit("contornos_grandes.csv tiene revisiones a mano escritas. No se escribió nada.")
    fotos = leer_particiones()
    assert len(fotos) == 974, f"se esperaban 974 fotos y hay {len(fotos)}"
    por_nombre = {f["archivo"]: f for f in fotos}
    DIR_MOSAICOS.mkdir(parents=True, exist_ok=True)

    medir_luz(fotos)

    elegidas = seleccionar_muestra2(fotos)
    escribir_csv(DIR_RES / "muestra2_seleccion.csv", ["indice", "archivo", "clase", "bloque"],
                 [{"indice": n, "archivo": f["archivo"], "clase": f["clase"], "bloque": f["bloque"]}
                  for n, f in enumerate(elegidas, start=1)])
    hojas_para_contar(elegidas)
    ruta_mano = DIR_RES / "muestra2_conteo_a_mano.csv"
    if not ruta_mano.exists():
        print(f"Paso 1 listo: {len(elegidas)} fotos en mosaicos\\muestra2_hoja_N.png. Falta "
              "resultados\\fase2\\muestra2_conteo_a_mano.csv (indice, archivo, clase, n_piezas_a_mano).")
        return
    mano = leer_csv(ruta_mano)
    if sorted(f["archivo"] for f in mano) != sorted(f["archivo"] for f in elegidas):
        raise SystemExit("El conteo a mano no corresponde a la muestra 2 seleccionada.")

    # Diagnóstico: la muestra 1 (ya vista) con los métodos nuevos. No decide nada.
    tabla1 = tabla_coincidencias(analizar_muestra(leer_csv(DIR_RES / "muestra_medicion.csv"), por_nombre))
    escribir_csv(DIR_RES / "medicion_muestra1.csv", list(tabla1[0].keys()), tabla1)

    muestra2 = analizar_muestra(mano, por_nombre)
    tabla = tabla_coincidencias(muestra2)
    escribir_csv(DIR_RES / "medicion_segmentacion.csv", list(tabla[0].keys()), tabla)
    for m in METODOS:
        mosaico_control(muestra2, m)
    metodo, aciertos = elegir_metodo(tabla)
    texto = ", ".join(f"({m}) {aciertos[m]}" for m in METODOS)
    print(f"2. Muestra 2 ({len(tabla)} fotos): coincidencias con el conteo a mano: {texto} -> gana ({metodo})")
    if aciertos[metodo] < MINIMO_ACIERTOS:
        print(f"   El ganador no llega a {MINIMO_ACIERTOS} de {len(tabla)}. Se detiene: no se escribe "
              "parametros.json ni se mide el filtro. Hay que decidir antes.")
        return

    parametros = {"segmentacion": metodo, "fondo": {"factor": FACTOR_FONDO, "ventana": VENTANA_FONDO},
                  "radio_min_px": RADIO_MIN_PX, "area_min_px": AREA_MIN_PX, "filtro": "mediana_3",
                  "muestra2_aciertos": aciertos}
    (DIR_RES / "parametros.json").write_text(json.dumps(parametros, indent=2) + "\n", encoding="utf-8")
    print(f"   parametros.json: {parametros}")
    medir_filtro(fotos, metodo)
    medir_contornos_grandes(fotos, metodo)


if __name__ == "__main__":
    main()
