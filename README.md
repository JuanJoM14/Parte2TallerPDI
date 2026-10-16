# Clasificación de piezas Lego con características diseñadas a mano

Parte 2 de la Tarea 2 de Procesamiento Digital de Imágenes (Universidad de Antioquia, 2026-2).

El sistema recibe una foto con piezas Lego, encuentra cada pieza, la clasifica en una de 4 clases (**Brick 1x4**, **Brick 2x2 L**, **Plate 2x2** y **Plate 2x4**) y muestra la confianza de cada clasificación. Usa procesamiento clásico de imágenes (filtro de mediana, umbral de Otsu, morfología), características diseñadas a mano (HOG, forma con momentos de Hu y textura LBP) y clasificadores de scikit-learn (SVM y Random Forest). Incluye una aplicación web en Streamlit.

**Resultado:** el modelo final (forma + LBP con SVM lineal) acierta las 133 piezas del conjunto de prueba. En la prueba de robustez con escenas de varias piezas y cambios de luz y fondo, clasifica bien el 98,7 % de las piezas que detecta; las fallas vienen de la segmentación (piezas claras junto a piezas oscuras).

---

## 1. Requisitos

- **Windows** (los comandos de este documento son para Windows; en Linux o macOS cambie `.venv\Scripts\python` por `.venv/bin/python` y `\` por `/`).
- **Python 3.12.**
- El banco de fotos de Kaggle (sección 3).

## 2. Instalación

Abra una terminal en la carpeta `Taller 2` (la que contiene este archivo) y ejecute:

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

`requirements.txt` instala OpenCV, NumPy, scikit-image, scikit-learn, joblib, matplotlib, pandas y Streamlit con las versiones usadas en el proyecto.

**Todos los comandos se corren desde la carpeta `Taller 2`.** Los scripts usan rutas relativas a esa carpeta, así que funcionan aunque la ruta tenga tildes o espacios.

## 3. Datos

Banco: **"Lego Brick Sorting - Image Recognition"**, del usuario pacogarciam3 en Kaggle:
https://www.kaggle.com/datasets/pacogarciam3/lego-brick-sorting-image-recognition/data

Descárguelo y deje dentro de `datos_originales\` esta estructura (solo se usan 4 de las 20 clases y solo las fotos base):

```
datos_originales\
├── ImageSetKey.csv
├── background_backlit_A.jpg
├── background_backlit_B.jpg
├── Base Images\
│   ├── Brick_1x4\         (238 fotos)
│   ├── Brick_2x2_L\       (317 fotos)
│   ├── Plate_2x2\         (207 fotos)
│   └── Plate_2x4\         (212 fotos)
└── Cropped Images\         (no se usa)
```

Las fotos originales nunca se modifican: todo lo que generan los scripts va a carpetas nuevas.

## 4. Uso rápido: abrir la aplicación

Si la carpeta ya trae `modelos\modelo_final.joblib` (el modelo entrenado), no hace falta entrenar nada:

```
.venv\Scripts\streamlit run Codigo\app.py
```

Se abre una página en el navegador. Suba una foto (JPG o PNG) con una o varias piezas sobre fondo claro. La aplicación muestra:

- la foto con un rectángulo por pieza y la etiqueta «clase confianza %»;
- el conteo de piezas por clase y el tiempo de procesamiento;
- una tabla con cada pieza (clase, confianza, área y si toca el borde de la foto).

Para probarla sirven las imágenes de `escenas\` (varias piezas por imagen) o cualquier foto de `datos_originales\Base Images\`.

La aplicación no entrena nada: carga el modelo guardado una vez y, por cada foto nueva, repite los pasos de preprocesamiento, segmentación y extracción de características (`Codigo\reconocer.py`). Tarda menos de un segundo por foto.

Para cerrarla, presione `Ctrl + C` en la terminal.

## 5. Reproducir el proyecto completo

Los scripts se corren en este orden. Cada uno lee lo que dejó el anterior.

| # | Fase | Comando | Qué produce | Tiempo aprox. |
|---|---|---|---|---|
| 1 | Partición | `.venv\Scripts\python Codigo\particion.py` | `particiones\`: `train.csv`, `val.csv`, `test.csv` y copias de las fotos por conjunto y clase | 1 min |
| 2 | Recortes (ROI) | `.venv\Scripts\python Codigo\generar_roi.py` | `roi\`: 951 recortes de 128×128 con su máscara y `metadatos.csv` | 2–5 min |
| 3 | Características | `.venv\Scripts\python Codigo\caracteristicas.py` | `caracteristicas\`: `train.npz`, `val.npz`, `test.npz` (8.139 columnas por pieza) | 1–3 min |
| 4 | Estudio de HOG | `.venv\Scripts\python Codigo\estudio_hog.py` | `resultados\fase3\estudio_hog.csv` | 5 min |
| 5 | Entrenamiento | `.venv\Scripts\python Codigo\entrenar.py` | `modelos\` (15 modelos y `modelo_final.joblib`) y `resultados\fase4\` | 3 min |
| 6 | Evaluación en prueba | `.venv\Scripts\python Codigo\evaluar.py` | `resultados\fase5\`: tabla de los 15 modelos, matrices de confusión, ROC, curva de aprendizaje | 30 s |
| 6b | Galería de errores | `.venv\Scripts\python Codigo\galeria_errores.py` | `resultados\fase5\galeria_errores_otros_modelos.png`: piezas de prueba que fallaron los modelos distintos al final (el final no falló ninguna) | 5 s |
| 7 | Escenas | `.venv\Scripts\python Codigo\generar_escenas.py` | `escenas\`: 10 escenas, 40 variantes de luz y fondo y `verdad.csv` | 10 s |
| 8 | Robustez | `.venv\Scripts\python Codigo\evaluar_escenas.py` | `resultados\fase6\`: aciertos y fallos por tipo de imagen e imágenes anotadas | 10 s |
| 9 | Aplicación | `.venv\Scripts\streamlit run Codigo\app.py` | Página web (sección 4) | — |

Todo proceso aleatorio usa la **semilla 42**, así que los resultados se repiten.

### Reglas que conviene saber

- **Los scripts no sobrescriben.** Cada uno se detiene si su salida ya existe (por ejemplo, `generar_roi.py` se detiene si existe `roi\metadatos.csv`). Para repetir un paso, mueva su carpeta de salida fuera de `Taller 2` y vuelva a correrlo; después repita los pasos siguientes.
- **La prueba se usa una sola vez.** `evaluar.py` es el «examen final» y se bloquea si `resultados\fase5\tabla_prueba.csv` ya existe. Los hiperparámetros y el modelo final se eligen solo con validación (`entrenar.py`).
- **Escalado y PCA se ajustan solo con entrenamiento** (van dentro de cada modelo guardado).

### Parámetros de segmentación (fase 2)

La segmentación usa los parámetros guardados en `resultados\fase2\parametros.json`, que `generar_roi.py` lee directamente. Se eligieron con `Codigo\medir_fase2.py`, que necesita dos revisiones hechas a mano y ya incluidas en `resultados\fase2\`:

- `muestra2_conteo_a_mano.csv`: número de piezas contado a mano en 60 fotos.
- `contornos_grandes.csv`: revisión a mano de los contornos sospechosos de ser dos piezas pegadas (las fotos marcadas «dos piezas» salen del banco).

Para reproducir el proyecto **no hace falta** correr `medir_fase2.py`; solo para repetir la elección del método.

`limpieza_calidad.py` (en la raíz) es un diagnóstico previo de calidad de las fotos; no excluye ninguna foto y no hace falta correrlo.

## 6. Cómo funciona el sistema

1. **Partición** (fase 1): 70 % entrenamiento, 15 % validación y 15 % prueba, por clase y por bloques de 10 minutos de captura, para que fotos casi iguales no queden en conjuntos distintos.
2. **Preprocesamiento y segmentación** (fase 2): lado mayor a 640 px, mediana 3×3, gris dividido por el fondo estimado, umbral de Otsu, apertura y cierre con núcleo elíptico de 5×5, contornos con área mínima de 150 px. Cada pieza se recorta en un cuadrado de 256 px centrado en ella y se reduce a 128×128 (conserva el tamaño real).
3. **Características** (fase 3): HOG (9 orientaciones, celdas 8×8, bloques 2×2: 8.100 valores), forma (área, perímetro, proporción, solidez y 7 momentos de Hu: 11 valores) y LBP uniforme en dos escalas (P = 8, R = 1 y P = 16, R = 2: 28 valores).
4. **Clasificación** (fase 4): cinco experimentos (E1 HOG; E2 HOG + forma; E3 LBP; E4 forma + LBP; E5 los tres) con SVM lineal, SVM RBF y Random Forest. Se elige por F1 macro en validación. **Modelo final: E4 (forma + LBP) con SVM lineal, C = 0,1.**
5. **Evaluación** (fase 5): los 15 modelos en prueba, una sola vez; reporte por clase, matrices de confusión, ROC y curva de aprendizaje.
6. **Aplicación y robustez** (fase 6): aplicación Streamlit y prueba con 10 escenas armadas con piezas de prueba y 4 variantes (oscura, clara, luz desigual y otro fondo).

## 7. Estructura de carpetas

```
Taller 2\
├── README.md                 este archivo
├── requirements.txt          bibliotecas y versiones
├── Codigo\
│   ├── preprocesamiento.py   funciones de lectura, filtro, segmentación y recorte (las usan los demás)
│   ├── particion.py          fase 1
│   ├── medir_fase2.py        fase 2: mediciones para elegir la segmentación (opcional, sección 5)
│   ├── generar_roi.py        fase 2: recortes
│   ├── caracteristicas.py    fase 3: HOG, forma y LBP
│   ├── estudio_hog.py        fase 3: estudio de parámetros de HOG
│   ├── entrenar.py           fase 4: entrenamiento y elección del modelo
│   ├── evaluar.py            fase 5: evaluación en prueba
│   ├── galeria_errores.py    fase 5: galería de errores de los otros modelos
│   ├── reconocer.py          fase 6: reconocimiento de una foto (lo usa la app)
│   ├── app.py                fase 6: aplicación Streamlit
│   ├── generar_escenas.py    fase 6: escenas y variantes
│   └── evaluar_escenas.py    fase 6: prueba de robustez
├── datos_originales\         banco de Kaggle (no se modifica)
├── metodologia\              metodología, bibliotecas y técnicas, limitaciones
├── particiones\              partición y su README
├── roi\                      recortes de 128×128 y máscaras
├── caracteristicas\          vectores por conjunto
├── modelos\                  modelos entrenados y modelo final
├── escenas\                  escenas de varias piezas
├── resultados\               tablas, gráficas y mediciones por fase
└── Parte 1\                  notebook de filtros (Parte 1 del taller)
```

## 8. Limitaciones conocidas

- **Piezas claras junto a una oscura:** con un solo umbral para toda la imagen, la pieza clara puede quedar como fondo y no detectarse.
- **Piezas que se tocan** pueden detectarse como una sola.
- **Tamaño:** el modelo final usa área y perímetro, así que depende de que la cámara esté a la misma distancia que en el banco.
- **Banco:** todas las fotos son del mismo montaje (mismo fondo y cámara, dos días de julio de 2018); no hay fotos propias.
- **Piezas cortadas por el borde de la foto:** la aplicación las marca; el modelo no se entrenó con ellas.

## 9. Más detalle

- `metodologia\metodologia.md`: qué se hizo en cada fase, con qué parámetros, por qué y con qué resultados.
- `metodologia\bibliotecas_y_tecnicas.md`: bibliotecas y técnicas de cada fase, con su justificación.
- `metodologia\limitaciones.md`: limitaciones e incumplimientos del enunciado.

## 10. Qué incluye este repositorio

- **Incluye:** la metodología (`metodologia\`), el código, la partición del banco (`particiones\`), los recortes (`roi\`), las características (`caracteristicas\`), las escenas (`escenas\`), todos los resultados (`resultados\`), el modelo final y los modelos de E4 (`modelos\`) y el notebook de la Parte 1.
- **No incluye:**
  - `datos_originales\`: se descarga de Kaggle (sección 3). Hace falta para correr los pasos 1, 2, 7 y 8 de la sección 5; la aplicación funciona sin él.
  - Los otros 12 modelos de `modelos\`, por tamaño (18 a 24 MB cada uno). Se regeneran con `Codigo\entrenar.py` (unos 3 minutos). `evaluar.py` los necesita, pero su resultado ya está en `resultados\fase5\`.
  - `.venv\`: se crea con la sección 2.
