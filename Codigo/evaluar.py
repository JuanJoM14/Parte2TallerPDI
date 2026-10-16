"""Fase 5: evaluación en prueba (examen final), una sola vez.

Qué hace (detalle en documentacion\\PlanesContextoDeFase\\Fase5_Contexto.md y Fase5_Plan.md):
  - Evalúa en caracteristicas\\test.npz los 15 modelos de modelos\\ (tabla resumen, sección 3.4).
  - Modelo final: reporte por clase (precisión, recall, F1; macro y ponderado), ROC y AUC.
  - Matrices de confusión sin normalizar y normalizadas del mejor modelo de cada experimento
    (elegido en validación en la fase 4; el de E4 es el modelo final).
  - Galería de errores del modelo final y del mejor de E1 (solo HOG). Si el modelo final no tiene
    errores, muestra sus 8 piezas de menor confianza.
  - Curva de aprendizaje con learning_curve sobre entrenamiento, validación cruzada de 5 grupos por
    bloque de 10 minutos. Validación y prueba no se usan en la curva.
  No cambia ningún modelo: después de esta ejecución no se ajusta nada.
Cómo se corre (desde Taller 2):  .venv\\Scripts\\python Codigo\\evaluar.py
Qué escribe: resultados\\fase5\\ (tabla_prueba.csv, predicciones_prueba.csv, reporte_modelo_final.csv y
.txt, matrices\\, galeria_modelo_final.png, galeria_E1.png, roc.png, curva_aprendizaje.png y .csv,
resumen.json). Se detiene si resultados\\fase5\\tabla_prueba.csv ya existe.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc, classification_report,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold, learning_curve
from sklearn.preprocessing import label_binarize

from preprocesamiento import RAIZ, escribir_csv, leer_csv, leer_png

DIR_CAR = RAIZ / "caracteristicas"
DIR_MOD = RAIZ / "modelos"
DIR_RES = RAIZ / "resultados" / "fase5"
DIR_MAT = DIR_RES / "matrices"
DIR_ROI = RAIZ / "roi" / "imagenes"
SEMILLA = 42
N_TEST, N_COLUMNAS = 133, 8139
CLASIFICADORES = ("svm_lineal", "svm_rbf", "random_forest")
NOMBRES_CLF = {"svm_lineal": "SVM lineal", "svm_rbf": "SVM RBF", "random_forest": "Random Forest"}
PIEZAS_GALERIA_SIN_ERRORES = 8


def cargar(uso):
    d = np.load(DIR_CAR / f"{uso}.npz")
    return d["X"], d["y"], d["nombres"]


def mejor_por_experimento(mejores):
    """Mejor modelo de cada experimento en validación, con la misma regla de la fase 4."""
    res = {}
    for m in mejores:
        clave = (-float(m["f1_macro_val"]), -float(m["accuracy_val"]), CLASIFICADORES.index(m["clasificador"]))
        if m["experimento"] not in res or clave < res[m["experimento"]][0]:
            res[m["experimento"]] = (clave, m)
    return {e: v[1] for e, v in sorted(res.items())}


def matrices(nombre, titulo, y, pred, clases):
    for normalizar, sufijo, fmt in ((None, "conteo", "d"), ("true", "normalizada", ".2f")):
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        ConfusionMatrixDisplay.from_predictions(y, pred, labels=clases, normalize=normalizar, cmap="Blues",
                                                values_format=fmt, ax=ax, colorbar=False)
        ax.set_title(f"{titulo}\n{'normalizada por clase real' if normalizar else 'sin normalizar'}")
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(DIR_MAT / f"{nombre}_{sufijo}.png", dpi=150)
        plt.close(fig)


def galeria(ruta, titulo, casos):
    if not casos:
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.axis("off")
        ax.text(0.5, 0.5, f"{titulo}\nSin piezas para mostrar", ha="center", va="center")
        fig.savefig(ruta, dpi=150)
        plt.close(fig)
        return
    cols = min(4, len(casos))
    filas = int(np.ceil(len(casos) / cols))
    fig, ejes = plt.subplots(filas, cols, figsize=(3.2 * cols, 3.6 * filas), squeeze=False)
    for ax in ejes.ravel():
        ax.axis("off")
    for ax, c in zip(ejes.ravel(), casos):
        img = leer_png(DIR_ROI / c["nombre"])
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        color = "#b03a2e" if c["real"] != c["predicha"] else "#2a6fb0"
        ax.set_title(f"Real: {c['real']}\nPredicha: {c['predicha']}\nProb.: {c['probabilidad']:.2f}",
                     fontsize=8.5, color=color)
    fig.suptitle(titulo)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def casos_galeria(y, pred, proba, nombres, solo_errores):
    conf = proba.max(axis=1)
    idx = np.where(y != pred)[0]
    if len(idx) == 0 and not solo_errores:
        idx = np.argsort(conf)[:PIEZAS_GALERIA_SIN_ERRORES]
    return [{"nombre": str(nombres[i]), "real": str(y[i]), "predicha": str(pred[i]),
             "probabilidad": float(conf[i])} for i in idx]


def curva_aprendizaje(modelo_final, x_tr, y_tr, n_tr):
    metadatos = {f["nombre_roi"]: f["bloque"] for f in leer_csv(RAIZ / "roi" / "metadatos.csv")}
    grupos = np.array([metadatos[str(n)] for n in n_tr])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    tamanos, f1_ent, f1_cv = learning_curve(clone(modelo_final), x_tr, y_tr, groups=grupos, cv=cv,
                                            train_sizes=np.linspace(0.1, 1.0, 10), scoring="f1_macro",
                                            shuffle=True, random_state=SEMILLA, n_jobs=-1)
    filas = [{"piezas_entrenamiento": int(t), "f1_entrenamiento_media": round(float(a.mean()), 4),
              "f1_entrenamiento_desv": round(float(a.std()), 4), "f1_validacion_cruzada_media": round(float(b.mean()), 4),
              "f1_validacion_cruzada_desv": round(float(b.std()), 4)} for t, a, b in zip(tamanos, f1_ent, f1_cv)]
    escribir_csv(DIR_RES / "curva_aprendizaje.csv", list(filas[0].keys()), filas)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for datos, nombre, color in ((f1_ent, "Entrenamiento", "#2a6fb0"), (f1_cv, "Validación cruzada (5 grupos por bloque)", "#e08a1e")):
        media, desv = datos.mean(axis=1), datos.std(axis=1)
        ax.plot(tamanos, media, "o-", color=color, label=nombre)
        ax.fill_between(tamanos, media - desv, media + desv, color=color, alpha=0.15)
    ax.set_xlabel("Piezas de entrenamiento")
    ax.set_ylabel("F1 macro")
    ax.set_title("Curva de aprendizaje del modelo final (solo entrenamiento)")
    ax.set_ylim(min(0.5, float(f1_cv.mean(axis=1).min()) - 0.05), 1.01)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DIR_RES / "curva_aprendizaje.png", dpi=150)
    plt.close(fig)
    return filas


def roc(y, proba, clases, ruta):
    yb = label_binarize(y, classes=clases)
    fig, ax = plt.subplots(figsize=(6, 5))
    aucs = {}
    for k, clase in enumerate(clases):
        fpr, tpr, _ = roc_curve(yb[:, k], proba[:, k])
        aucs[clase] = round(float(auc(fpr, tpr)), 4)
        ax.plot(fpr, tpr, label=f"{clase} (AUC {aucs[clase]:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("ROC uno contra todos del modelo final (prueba)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    aucs["macro"] = round(float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=clases)), 4)
    return aucs


def main():
    if (DIR_RES / "tabla_prueba.csv").exists():
        raise SystemExit("resultados\\fase5\\tabla_prueba.csv ya existe: la prueba ya se usó y no se repite.")
    mejores = leer_csv(RAIZ / "resultados" / "fase4" / "mejores.csv")
    info_final = json.loads((DIR_MOD / "modelo_final.json").read_text(encoding="utf-8"))
    rutas = [DIR_MOD / m["modelo"] for m in mejores] + [DIR_MOD / "modelo_final.joblib"]
    if len(mejores) != 15 or not all(r.exists() for r in rutas):
        raise SystemExit("Faltan modelos de la fase 4.")
    fechas_antes = {r.name: r.stat().st_mtime_ns for r in rutas}

    x_te, y_te, n_te = cargar("test")
    if x_te.shape != (N_TEST, N_COLUMNAS):
        raise SystemExit(f"Forma inesperada de prueba: {x_te.shape}")
    clases = sorted(set(y_te.tolist()))
    DIR_MAT.mkdir(parents=True, exist_ok=True)

    # 1. Tabla resumen de las 15 combinaciones y predicciones por pieza
    tabla, predicciones, salidas = [], [], {}
    for m in mejores:
        modelo = joblib.load(DIR_MOD / m["modelo"])
        pred = modelo.predict(x_te)
        proba = modelo.predict_proba(x_te)
        salidas[m["modelo"]] = (pred, proba, list(modelo.classes_))
        acc, f1 = accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro")
        tabla.append({"experimento": m["experimento"], "descriptores": m["descriptores"],
                      "clasificador": m["clasificador"], "parametros": m["parametros"],
                      "accuracy_val": m["accuracy_val"], "f1_macro_val": m["f1_macro_val"],
                      "accuracy_prueba": round(acc, 4), "f1_macro_prueba": round(f1, 4),
                      "errores_prueba": int((pred != y_te).sum()), "es_modelo_final": m["es_modelo_final"]})
        for i in range(N_TEST):
            predicciones.append({"modelo": m["modelo"], "nombre_roi": str(n_te[i]), "clase_real": str(y_te[i]),
                                 "clase_predicha": str(pred[i]), "probabilidad_predicha": round(float(proba[i].max()), 4),
                                 "acierto": int(pred[i] == y_te[i])})
        print(f"{m['experimento']} {m['clasificador']}: prueba F1 {f1:.4f} acc {acc:.4f}")
    escribir_csv(DIR_RES / "tabla_prueba.csv", list(tabla[0].keys()), tabla)
    escribir_csv(DIR_RES / "predicciones_prueba.csv", list(predicciones[0].keys()), predicciones)

    # 2. Modelo final: reporte por clase y ROC
    final = joblib.load(DIR_MOD / "modelo_final.joblib")
    pred_f, proba_f = final.predict(x_te), final.predict_proba(x_te)
    if list(final.classes_) != clases:
        raise SystemExit("Las clases del modelo final no coinciden con las de prueba.")
    titulo_final = f"Modelo final: {info_final['experimento']} ({info_final['descriptores']}), {NOMBRES_CLF[info_final['clasificador']]}"
    texto = classification_report(y_te, pred_f, labels=clases, digits=4, zero_division=0)
    (DIR_RES / "reporte_modelo_final.txt").write_text(f"{titulo_final}\n\n{texto}", encoding="utf-8")
    rep = classification_report(y_te, pred_f, labels=clases, output_dict=True, zero_division=0)
    filas_rep = [{"clase": k, "precision": round(v["precision"], 4), "recall": round(v["recall"], 4),
                  "f1": round(v["f1-score"], 4), "piezas": int(v["support"])}
                 for k, v in rep.items() if isinstance(v, dict)]
    escribir_csv(DIR_RES / "reporte_modelo_final.csv", list(filas_rep[0].keys()), filas_rep)
    aucs = roc(y_te, proba_f, clases, DIR_RES / "roc.png")

    # 3. Matrices de confusión del mejor modelo de cada experimento
    por_exp = mejor_por_experimento(mejores)
    for exp, m in por_exp.items():
        pred = salidas[m["modelo"]][0]
        es_final = m["modelo"] == info_final["modelo"]
        titulo = f"{exp} ({m['descriptores']}), {NOMBRES_CLF[m['clasificador']]}" + (" - modelo final" if es_final else "")
        matrices(f"{exp}_{m['clasificador']}", titulo, y_te, pred, clases)

    # 4. Galerías
    casos_f = casos_galeria(y_te, pred_f, proba_f, n_te, solo_errores=False)
    hay_errores_final = bool((pred_f != y_te).any())
    galeria(DIR_RES / "galeria_modelo_final.png",
            titulo_final + ("" if hay_errores_final else f" - sin errores: {len(casos_f)} piezas de menor confianza"),
            casos_f)
    m_e1 = por_exp["E1"]
    pred_e1, proba_e1, _ = salidas[m_e1["modelo"]]
    casos_e1 = casos_galeria(y_te, pred_e1, proba_e1, n_te, solo_errores=True)
    galeria(DIR_RES / "galeria_E1.png", f"Errores de E1 (HOG), {NOMBRES_CLF[m_e1['clasificador']]}", casos_e1)

    # 5. Curva de aprendizaje (solo entrenamiento)
    x_tr, y_tr, n_tr = cargar("train")
    filas_curva = curva_aprendizaje(final, x_tr, y_tr, n_tr)

    # 6. Verificación
    fechas_despues = {r.name: r.stat().st_mtime_ns for r in rutas}
    fila_final = next(t for t in tabla if t["es_modelo_final"] in ("1", 1))
    if fechas_antes != fechas_despues:
        raise SystemExit("Algún modelo cambió durante la evaluación.")
    if round(accuracy_score(y_te, pred_f), 4) != fila_final["accuracy_prueba"]:
        raise SystemExit("Las cifras del modelo final no coinciden entre la tabla y el reporte.")
    resumen = {"modelo_final": {k: info_final[k] for k in ("experimento", "descriptores", "clasificador", "parametros")},
               "accuracy_prueba": fila_final["accuracy_prueba"], "f1_macro_prueba": fila_final["f1_macro_prueba"],
               "errores_prueba": fila_final["errores_prueba"], "auc": aucs,
               "galeria_modelo_final": "errores" if hay_errores_final else "sin errores; piezas de menor confianza",
               "errores_E1": len(casos_e1), "curva_aprendizaje_ultimo_punto": filas_curva[-1],
               "matrices": [f"{e}_{m['clasificador']}" for e, m in por_exp.items()],
               "modelos_sin_cambios": True}
    (DIR_RES / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modelo final en prueba: accuracy {fila_final['accuracy_prueba']}, F1 macro {fila_final['f1_macro_prueba']}, "
          f"errores {fila_final['errores_prueba']}")
    print("Verificación correcta. Listo.")


if __name__ == "__main__":
    main()
