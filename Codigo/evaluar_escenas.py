"""Prueba de robustez de la fase 6: pasa reconocer.detectar por las 50 imágenes de escenas\\.

Qué hace (detalle en documentacion\\PlanesContextoDeFase\\Fase6_Contexto.md, filas 10 y 11):
  - Comprobación inicial: en 5 fotos de prueba de una pieza, el vector que calcula reconocer.py
    debe ser igual al de caracteristicas\\test.npz y la clase igual a la de la fase 5.
  - Para cada imagen compara las cajas detectadas con escenas\\verdad.csv: una pieza está
    detectada si una caja la cubre con IoU >= 0,5, y bien clasificada si además la clase coincide.
    Cuenta también piezas perdidas y detecciones falsas.
  - No ajusta nada: si algo falla, se reporta.
Cómo se corre (desde Taller 2):  .venv\\Scripts\\python Codigo\\evaluar_escenas.py
Qué escribe: resultados\\fase6\\ (escenas.csv, resumen.csv, comprobacion.csv) y
resultados\\fase6\\anotadas\\ (las 50 imágenes con los rectángulos). Se detiene si
resultados\\fase6\\resumen.csv ya existe.
"""
from collections import defaultdict

import cv2
import numpy as np

from preprocesamiento import RAIZ, decodificar, escribir_csv, guardar_png, leer_csv, ruta_foto
from reconocer import cargar_modelo, detectar, dibujar, info_modelo, vector_pieza
from preprocesamiento import filtrar, normalizar, recorte_roi, segmentar

DIR_ESC = RAIZ / "escenas"
DIR_RES = RAIZ / "resultados" / "fase6"
DIR_ANOT = DIR_RES / "anotadas"
IOU_MIN = 0.5
TIPOS = ("original", "oscura", "clara", "luz_desigual", "otro_fondo")
N_COMPROBACION = 5


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter) if inter else 0.0


def comprobacion(modelo):
    """reconocer.py debe reproducir el vector de la fase 3 y la clase de la fase 5."""
    datos = np.load(RAIZ / "caracteristicas" / "test.npz")
    indice = {str(n): i for i, n in enumerate(datos["nombres"])}
    pred5 = {f["nombre_roi"]: f["clase_predicha"]
             for f in leer_csv(RAIZ / "resultados" / "fase5" / "predicciones_prueba.csv")
             if f["modelo"] == info_modelo()["modelo"]}
    filas = [f for f in leer_csv(RAIZ / "roi" / "metadatos.csv") if f["uso"] == "test" and f["n_piezas"] == "1"]
    filas.sort(key=lambda f: f["nombre_roi"])
    paso = max(1, len(filas) // N_COMPROBACION)
    salida = []
    for f in filas[::paso][:N_COMPROBACION]:
        img = filtrar(normalizar(decodificar(ruta_foto(f["clase"], f["archivo"]))))
        p = segmentar(img, "b1")[0]
        vec = vector_pieza(*recorte_roi(img, p["mascara"], p["caja"]))
        igual = bool(np.allclose(vec, datos["X"][indice[f["nombre_roi"]]], atol=1e-4))
        _, piezas, _ = detectar(decodificar(ruta_foto(f["clase"], f["archivo"])), modelo)
        clase = piezas[0]["clase"] if piezas else ""
        salida.append({"nombre_roi": f["nombre_roi"], "vector_igual_fase3": int(igual),
                       "clase_fase5": pred5[f["nombre_roi"]], "clase_reconocer": clase,
                       "clase_igual": int(clase == pred5[f["nombre_roi"]])})
    escribir_csv(DIR_RES / "comprobacion.csv", list(salida[0].keys()), salida)
    if not all(s["vector_igual_fase3"] and s["clase_igual"] for s in salida):
        raise SystemExit("reconocer.py no reproduce las fases 3 y 5; revisar resultados\\fase6\\comprobacion.csv")
    print(f"Comprobación correcta en {len(salida)} fotos de prueba.")


def main():
    if (DIR_RES / "resumen.csv").exists():
        raise SystemExit("resultados\\fase6\\resumen.csv ya existe; moverlo para repetir.")
    if not (DIR_ESC / "verdad.csv").exists():
        raise SystemExit("Falta escenas\\verdad.csv: correr antes Codigo\\generar_escenas.py.")
    DIR_ANOT.mkdir(parents=True, exist_ok=True)
    modelo = cargar_modelo()
    comprobacion(modelo)

    verdad = defaultdict(list)
    for v in leer_csv(DIR_ESC / "verdad.csv"):
        verdad[(v["imagen"], v["tipo"])].append(v)
    filas, tiempos = [], defaultdict(list)
    for (imagen, tipo), reales in sorted(verdad.items()):
        img, piezas, ms = detectar(decodificar(DIR_ESC / imagen), modelo)
        tiempos[tipo].append(ms)
        usadas = set()
        for r in reales:
            caja_r = (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
            mejor, mejor_iou = None, 0.0
            for j, p in enumerate(piezas):
                if j not in usadas and iou(caja_r, p["caja"]) > mejor_iou:
                    mejor, mejor_iou = j, iou(caja_r, p["caja"])
            fila = {"imagen": imagen, "tipo": tipo, "clase_real": r["clase"], "origen": r["origen"]}
            if mejor is not None and mejor_iou >= IOU_MIN:
                usadas.add(mejor)
                p = piezas[mejor]
                estado = "bien_clasificada" if p["clase"] == r["clase"] else "mal_clasificada"
                fila.update(clase_predicha=p["clase"], confianza=round(p["confianza"], 4), iou=round(mejor_iou, 3),
                            estado=estado)
            else:
                fila.update(clase_predicha="", confianza="", iou=round(mejor_iou, 3), estado="perdida")
            filas.append(fila)
        for j, p in enumerate(piezas):
            if j not in usadas:
                filas.append({"imagen": imagen, "tipo": tipo, "clase_real": "", "origen": "",
                              "clase_predicha": p["clase"], "confianza": round(p["confianza"], 4), "iou": "",
                              "estado": "deteccion_falsa"})
        anotada = dibujar(img, piezas)
        for r in reales:  # cajas reales en gris fino, para ver piezas perdidas
            x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
            cv2.rectangle(anotada, (x - 3, y - 3), (x + w + 3, y + h + 3), (120, 120, 120), 1)
        guardar_png(DIR_ANOT / imagen, anotada)

    escribir_csv(DIR_RES / "escenas.csv", list(filas[0].keys()), filas)
    resumen = []
    for tipo in TIPOS:
        f = [x for x in filas if x["tipo"] == tipo]
        reales = [x for x in f if x["estado"] != "deteccion_falsa"]
        n = len(reales)
        cuenta = lambda e: sum(x["estado"] == e for x in f)
        bien, mal, perd = cuenta("bien_clasificada"), cuenta("mal_clasificada"), cuenta("perdida")
        resumen.append({"tipo": tipo, "imagenes": len(tiempos[tipo]), "piezas": n, "detectadas": bien + mal,
                        "bien_clasificadas": bien, "mal_clasificadas": mal, "perdidas": perd,
                        "detecciones_falsas": cuenta("deteccion_falsa"),
                        "porcentaje_bien": round(100 * bien / n, 1) if n else "",
                        "tiempo_medio_ms": round(float(np.mean(tiempos[tipo])), 1)})
        print(f"{tipo:13s} piezas {n:3d}  bien {bien:3d}  mal {mal:2d}  perdidas {perd:2d}  "
              f"falsas {cuenta('deteccion_falsa'):2d}")
    escribir_csv(DIR_RES / "resumen.csv", list(resumen[0].keys()), resumen)
    print("Listo.")


if __name__ == "__main__":
    main()
