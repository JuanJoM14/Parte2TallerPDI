"""Fase 4: entrena los clasificadores y elige hiperparámetros y modelo final con validación.

Qué hace (detalle en documentacion\\PlanesContextoDeFase\\Fase4_Contexto.md y Fase4_Plan.md):
  - Carga caracteristicas\\train.npz y val.npz. test.npz NO se abre en esta fase.
  - Cinco experimentos: E1 HOG; E2 HOG + forma; E3 LBP; E4 forma + LBP; E5 HOG + forma + LBP.
  - Tres clasificadores: SVM lineal, SVM RBF y Random Forest.
  - Cada modelo es un Pipeline: escalado por columna (y PCA al 95 % solo sobre HOG), ajustado solo
    con entrenamiento, seguido del clasificador. Todos reciben el vector completo de 8.139 columnas
    y se quedan con las de su experimento, así la aplicación les pasa siempre el mismo vector.
  - Búsqueda en cuadrícula: cada combinación se entrena con entrenamiento y se mide en validación.
    Se elige por F1 macro; desempates: accuracy, después el orden de la cuadrícula (de lo más
    simple a lo más complejo).
  - Modelo final: el mejor de los 15; desempates: F1 macro, accuracy, menos columnas, clasificador
    más simple (SVM lineal, SVM RBF, Random Forest).
  - Gráficas: varianza acumulada del PCA de HOG, importancia de características (Random Forest
    sobre E5 sin PCA) y PCA 2D de E5.
Cómo se corre (desde Taller 2):  .venv\\Scripts\\python Codigo\\entrenar.py
Qué escribe: modelos\\ (15 modelos, modelo_final.joblib y modelo_final.json) y resultados\\fase4\\
(busqueda.csv, mejores.csv, resumen.json, importancia_bloques.csv, importancia_top20.csv,
varianza_pca.png, importancia.png y pca_2d.png). Se detiene si resultados\\fase4\\mejores.csv ya existe.
"""
import itertools
import json
import shutil
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocesamiento import RAIZ, escribir_csv

DIR_CAR = RAIZ / "caracteristicas"
DIR_MOD = RAIZ / "modelos"
DIR_RES = RAIZ / "resultados" / "fase4"
SEMILLA = 42
VARIANZA_PCA = 0.95
N_TRAIN, N_VAL, N_COLUMNAS = 674, 144, 8139

EXPERIMENTOS = {
    "E1": ("hog",),
    "E2": ("hog", "forma"),
    "E3": ("lbp",),
    "E4": ("forma", "lbp"),
    "E5": ("hog", "forma", "lbp"),
}
CLASIFICADORES = ("svm_lineal", "svm_rbf", "random_forest")  # de lo más simple a lo más complejo
# Cuadrículas en orden de lo más simple (más regularizado) a lo más complejo
CUADRICULAS = {
    "svm_lineal": [{"C": c} for c in (0.01, 0.1, 1, 10, 100)],
    "svm_rbf": [{"C": c, "gamma": g} for c, g in
                itertools.product((0.1, 1, 10, 100), (0.0001, 0.001, 0.01, "scale"))],
    "random_forest": [{"n_estimators": n, "max_depth": d} for n, d in
                      itertools.product((200, 500), (10, 20, None))],
}


def cargar(uso):
    d = np.load(DIR_CAR / f"{uso}.npz")
    return d["X"], d["y"], [str(c) for c in d["columnas"]]


def indices_por_bloque(columnas):
    bloques = {"hog": [], "forma": [], "lbp": []}
    for i, c in enumerate(columnas):
        if c.startswith("hog"):
            bloques["hog"].append(i)
        elif c.startswith("lbp"):
            bloques["lbp"].append(i)
        else:
            bloques["forma"].append(i)
    return bloques


def clasificador(nombre, params):
    # La SVM no da probabilidades por sí sola. Se calibran con el método sigmoide (Platt) en 5 partes
    # del entrenamiento; es el reemplazo que recomienda scikit-learn 1.9 para SVC(probability=True).
    if nombre in ("svm_lineal", "svm_rbf"):
        kernel = "linear" if nombre == "svm_lineal" else "rbf"
        return CalibratedClassifierCV(SVC(kernel=kernel, random_state=SEMILLA, **params),
                                      method="sigmoid", cv=5, ensemble=False)
    return RandomForestClassifier(max_features="sqrt", random_state=SEMILLA, n_jobs=-1, **params)


def armar_modelo(exp, nombre, params, bloques):
    partes = []
    for b in EXPERIMENTOS[exp]:
        if b == "hog":
            paso = Pipeline([("escala", StandardScaler()),
                             ("pca", PCA(n_components=VARIANZA_PCA, svd_solver="full",
                                         random_state=SEMILLA))])
        else:
            paso = StandardScaler()
        partes.append((b, paso, bloques[b]))
    prep = ColumnTransformer(partes, remainder="drop")
    return Pipeline([("prep", prep), ("clf", clasificador(nombre, params))])


def n_columnas(exp, bloques):
    return sum(len(bloques[b]) for b in EXPERIMENTOS[exp])


def componentes_pca(modelo):
    prep = modelo.named_steps["prep"]
    if "hog" not in prep.named_transformers_:
        return ""
    return int(prep.named_transformers_["hog"].named_steps["pca"].n_components_)


def medir(modelo, x, y):
    pred = modelo.predict(x)
    return accuracy_score(y, pred), f1_score(y, pred, average="macro")


def grafica_varianza(x_tr, bloques):
    pca = PCA(svd_solver="full").fit(StandardScaler().fit_transform(x_tr[:, bloques["hog"]]))
    acum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(acum, VARIANZA_PCA) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(acum) + 1), acum, color="#2a6fb0")
    ax.axhline(VARIANZA_PCA, color="#b03a2e", ls="--", lw=1, label=f"{VARIANZA_PCA:.0%}")
    ax.axvline(k, color="#555555", ls=":", lw=1, label=f"{k} componentes")
    ax.set_xlabel("Número de componentes")
    ax.set_ylabel("Varianza acumulada explicada")
    ax.set_title("PCA sobre HOG (entrenamiento)")
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(DIR_RES / "varianza_pca.png", dpi=150)
    plt.close(fig)
    return k


def importancia(x_tr, y_tr, columnas, bloques, params_rf):
    idx = bloques["hog"] + bloques["forma"] + bloques["lbp"]
    rf = RandomForestClassifier(max_features="sqrt", random_state=SEMILLA, n_jobs=-1, **params_rf)
    rf.fit(x_tr[:, idx], y_tr)
    imp = rf.feature_importances_
    nombres = [columnas[i] for i in idx]
    por_bloque, inicio = [], 0
    for b in ("hog", "forma", "lbp"):
        n = len(bloques[b])
        por_bloque.append({"bloque": b, "columnas": n,
                           "importancia_total": round(float(imp[inicio:inicio + n].sum()), 4),
                           "importancia_media_por_columna": round(float(imp[inicio:inicio + n].mean()), 6)})
        inicio += n
    orden = np.argsort(imp)[::-1][:20]
    top = [{"puesto": k + 1, "columna": nombres[i], "importancia": round(float(imp[i]), 5)}
           for k, i in enumerate(orden)]
    escribir_csv(DIR_RES / "importancia_bloques.csv", list(por_bloque[0].keys()), por_bloque)
    escribir_csv(DIR_RES / "importancia_top20.csv", list(top[0].keys()), top)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 2]})
    a1.bar([p["bloque"] for p in por_bloque], [p["importancia_total"] for p in por_bloque],
           color="#2a6fb0")
    a1.set_title("Importancia total por bloque")
    a2.barh([t["columna"] for t in top][::-1], [t["importancia"] for t in top][::-1], color="#2a6fb0")
    a2.set_title("20 columnas más importantes")
    fig.suptitle("Random Forest sobre E5 sin PCA (entrenamiento)")
    fig.tight_layout()
    fig.savefig(DIR_RES / "importancia.png", dpi=150)
    plt.close(fig)
    return por_bloque


def grafica_2d(x_tr, y_tr, bloques):
    idx = bloques["hog"] + bloques["forma"] + bloques["lbp"]
    z = PCA(n_components=2, random_state=SEMILLA).fit_transform(StandardScaler().fit_transform(x_tr[:, idx]))
    colores = ["#2a6fb0", "#e08a1e", "#3a9a5b", "#b03a2e"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for c, clase in zip(colores, sorted(set(y_tr))):
        m = y_tr == clase
        ax.scatter(z[m, 0], z[m, 1], s=12, alpha=0.7, color=c, label=clase)
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.set_title("E5 en 2D con PCA (entrenamiento)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DIR_RES / "pca_2d.png", dpi=150)
    plt.close(fig)


def main():
    if (DIR_RES / "mejores.csv").exists():
        raise SystemExit("resultados\\fase4\\mejores.csv ya existe; mover resultados\\fase4 y modelos para repetir.")
    x_tr, y_tr, columnas = cargar("train")
    x_va, y_va, columnas_va = cargar("val")
    if x_tr.shape != (N_TRAIN, N_COLUMNAS) or x_va.shape != (N_VAL, N_COLUMNAS) or columnas != columnas_va:
        raise SystemExit(f"Formas inesperadas: train {x_tr.shape}, val {x_va.shape}.")
    bloques = indices_por_bloque(columnas)
    DIR_RES.mkdir(parents=True, exist_ok=True)
    DIR_MOD.mkdir(exist_ok=True)

    k95 = grafica_varianza(x_tr, bloques)
    print(f"PCA de HOG: {k95} componentes para el {VARIANZA_PCA:.0%} de la varianza")

    busqueda, mejores = [], []
    for exp in EXPERIMENTOS:
        for nombre in CLASIFICADORES:
            mejor = None
            for orden, params in enumerate(CUADRICULAS[nombre]):
                t0 = time.perf_counter()
                modelo = armar_modelo(exp, nombre, params, bloques).fit(x_tr, y_tr)
                acc, f1 = medir(modelo, x_va, y_va)
                fila = {"experimento": exp, "clasificador": nombre, "orden_cuadricula": orden,
                        "parametros": json.dumps(params), "accuracy_val": round(acc, 4),
                        "f1_macro_val": round(f1, 4), "componentes_pca": componentes_pca(modelo),
                        "segundos": round(time.perf_counter() - t0, 2)}
                busqueda.append(fila)
                # mejor F1; desempate por accuracy; después gana el primero de la cuadrícula
                if mejor is None or (f1, acc) > (mejor[0], mejor[1]):
                    mejor = (f1, acc, params, modelo, fila)
            f1, acc, params, modelo, fila = mejor
            ruta = DIR_MOD / f"{exp}_{nombre}.joblib"
            joblib.dump(modelo, ruta)
            mejores.append({"experimento": exp, "descriptores": " + ".join(EXPERIMENTOS[exp]),
                            "clasificador": nombre, "parametros": json.dumps(params),
                            "columnas": n_columnas(exp, bloques), "componentes_pca": fila["componentes_pca"],
                            "accuracy_val": round(acc, 4), "f1_macro_val": round(f1, 4),
                            "errores_val": int(round((1 - acc) * N_VAL)), "modelo": ruta.name})
            print(f"{exp} {nombre}: F1 {f1:.4f}  acc {acc:.4f}  {params}")

    clave = lambda m: (-m["f1_macro_val"], -m["accuracy_val"], m["columnas"],
                       CLASIFICADORES.index(m["clasificador"]))
    final = sorted(mejores, key=clave)[0]
    for m in mejores:
        m["es_modelo_final"] = int(m is final)
    escribir_csv(DIR_RES / "busqueda.csv", list(busqueda[0].keys()), busqueda)
    escribir_csv(DIR_RES / "mejores.csv", list(mejores[0].keys()), mejores)
    shutil.copyfile(DIR_MOD / final["modelo"], DIR_MOD / "modelo_final.joblib")
    info_final = {k: v for k, v in final.items() if k != "es_modelo_final"}
    info_final["parametros"] = json.loads(final["parametros"])
    info_final["entrada"] = f"vector completo de {N_COLUMNAS} columnas en el orden de caracteristicas\\*.npz"
    info_final["clases"] = sorted(set(y_tr.tolist()))
    (DIR_MOD / "modelo_final.json").write_text(json.dumps(info_final, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    print(f"Modelo final: {final['experimento']} {final['clasificador']}")

    # Verificación: los modelos guardados se cargan, dan las mismas cifras y solo vieron entrenamiento
    for m in mejores:
        modelo = joblib.load(DIR_MOD / m["modelo"])
        acc, f1 = medir(modelo, x_va, y_va)
        proba = modelo.predict_proba(x_va)
        escalas = [t[1] if isinstance(t[1], StandardScaler) else t[1].named_steps["escala"]
                   for t in modelo.named_steps["prep"].transformers_ if t[0] != "remainder"]
        if (round(acc, 4), round(f1, 4)) != (m["accuracy_val"], m["f1_macro_val"]) \
                or not np.allclose(proba.sum(axis=1), 1) \
                or any(int(e.n_samples_seen_) != N_TRAIN for e in escalas):
            raise SystemExit(f"Falla la verificación de {m['modelo']}")

    rf_e5 = next(m for m in mejores if m["experimento"] == "E5" and m["clasificador"] == "random_forest")
    por_bloque = importancia(x_tr, y_tr, columnas, bloques, json.loads(rf_e5["parametros"]))
    grafica_2d(x_tr, y_tr, bloques)

    resumen = {"componentes_pca_hog_95": k95, "combinaciones_probadas": len(busqueda),
               "modelo_final": info_final, "importancia_por_bloque": por_bloque,
               "test_npz_abierto": False}
    (DIR_RES / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Verificación correcta. Listo.")


if __name__ == "__main__":
    main()
