# Bibliotecas y técnicas por fase

Bibliotecas y técnicas que usa cada fase del proyecto, con una justificación breve. Las versiones exactas están en `requirements.txt`; las decisiones y los resultados, en `metodologia.md`.

## Fase 1: Banco y partición

### Bibliotecas

Todas son de la biblioteca estándar de Python; la fase no necesita instalar nada.

| Biblioteca | Uso en la fase | Justificación |
|---|---|---|
| `csv` | Lee `ImageSetKey.csv` (con `utf-8-sig`) y escribe `train.csv`, `val.csv` y `test.csv`, con las filas en orden fijo | Formato simple y legible. Con 974 filas no hace falta otra biblioteca de tablas |
| `re` | Valida el nombre `2_<clase>_AAMMDDHHMMSS.jpg` y extrae la clase y la fecha | El código toma la hora de captura solo del nombre del archivo |
| `random` | `Random(42)` mezcla el orden de los bloques de cada clase | Con semilla fija y un generador por clase, el reparto se reproduce y no depende del orden de las clases |
| `shutil` | `copy2` copia cada foto a `particiones\<conjunto>\<clase>\` | Conserva la fecha de modificación y otros metadatos del original |
| `collections` | `defaultdict` agrupa las fotos por bloque y cuenta las copias por carpeta | Agrupa sin comprobar antes si la clave ya existe |
| `pathlib` | Calcula las rutas de `datos_originales\` y `particiones\` desde la ubicación del script | El código no tiene rutas fijas del computador |

### Técnicas

| Técnica | Uso en la fase | Justificación |
|---|---|---|
| Bloques de 10 minutos | Agrupa las fotos de una clase cuyo nombre cae en el mismo intervalo. Cada bloque va entero a un conjunto | Las fotos seguidas están a segundos de distancia y son casi iguales. Separarlas dejaría en prueba fotos casi idénticas a otras de entrenamiento (fuga de datos) |
| Partición estratificada | Cada clase se reparte por separado | Las cuatro clases quedan en los tres conjuntos, con 70/15/15 dentro de cada una |
| Reparto por déficit | Cada bloque, en orden, va al conjunto con mayor diferencia entre su objetivo y lo que ya tiene, medido en fotos. Empates: entrenamiento, validación y prueba | Los bloques no se pueden partir. Asignar cada uno al conjunto más atrasado mantiene las proporciones cerca del objetivo |
| Mezcla con semilla fija | Los bloques de cada clase se ordenan cronológicamente y luego se mezclan con semilla 42 | El orden de los días no decide qué queda en prueba, y el resultado se reproduce |
| Mediana de intervalos | Tiempo entre fotos seguidas de una misma clase: entre 13 s y 23 s según la clase. Es un análisis previo, no está en `Codigo\` | Justifica el bloque de 10 minutos. La mediana no cambia por los saltos de horas o días entre sesiones |
| Cruce entre lista y carpetas | Comprueba que no falte ni sobre ninguna foto frente a `ImageSetKey.csv`, y que cada nombre siga el patrón y tenga la clase correcta | Garantiza que el banco es exactamente la lista de fotos del dataset |
| Guardas de ejecución | Antes de escribir, comprueba la lista, las proporciones, los bloques, el total y las cifras esperadas de la partición. Si `particiones\` ya existe, se detiene sin tocarla | Evita escribir una partición que no cumple las reglas y no sobrescribe una ya generada |
| Verificación independiente | Recalcula la partición desde `ImageSetKey.csv` con un script de revisión aparte, y compara CSV, copias y tamaños de archivo | Las comprobaciones internas cuentan archivos por carpeta. La verificación independiente también revisa nombres y tamaños |

## Fase 2: Preprocesamiento y segmentación

### Bibliotecas

OpenCV y NumPy; las demás son de la biblioteca estándar.

| Biblioteca | Uso en la fase | Justificación |
|---|---|---|
| `cv2` (OpenCV) | Lectura y escritura de imágenes, filtro, umbral de Otsu, morfología, contornos y redimensionado | Tiene todas las operaciones que pide el enunciado (sección 3.2) |
| `numpy` | Lectura de archivos con `np.fromfile`, máscaras, distancias al centro y división por el fondo | OpenCV trabaja con arreglos de NumPy |
| `json` | Lee y escribe `resultados\fase2\parametros.json` | Los parámetros elegidos quedan en un archivo y `generar_roi.py` no los repite en el código |
| `csv` | Tablas de medición, `metadatos.csv` y listas de control | Formato simple y legible |
| `random` | `Random(42)` elige la muestra de 60 fotos para el conteo a mano y los recortes de los mosaicos de control | Con semilla fija, el resultado se reproduce |
| `collections`, `pathlib` | Conteos por foto y rutas relativas a `Taller 2` | El código no tiene rutas fijas del computador |

### Técnicas

| Técnica | Uso en la fase | Justificación |
|---|---|---|
| Lectura con `np.fromfile` y `cv2.imdecode` | Abre las fotos; los PNG se guardan con `cv2.imencode` y `tofile` | La ruta tiene acentos y `cv2.imread` falla con ellos en Windows |
| Normalización a lado mayor de 640 px con `INTER_AREA` | Primer paso de cada foto | Lo pide el enunciado. `INTER_AREA` es la interpolación indicada para reducir |
| Mediana 3×3 | Sobre la imagen BGR, antes de segmentar | Mejor configuración fija en el notebook de la Parte 1 y la que menos altera los bordes (`metodologia.md`, sección 2) |
| División por el fondo | Gris dividido por su fondo estimado (mediana de 63 px sobre la foto reducida 4 veces). Solo para segmentar | Quita las variaciones de luz dentro de la foto antes del umbral |
| Umbral de Otsu con polaridad por esquinas | Umbral global; la pieza queda del lado contrario a las cuatro esquinas de 20×20 px | Otsu no necesita fijar un umbral a mano. Las esquinas casi siempre son fondo |
| Apertura y cierre con kernel elíptico 5×5 | Limpia la máscara | Lo pide el enunciado. Quita puntos sueltos y cierra huecos pequeños |
| Contornos externos, área mínima 150 px y grosor mínimo 9 px | Separa las piezas y descarta ruido y franjas de sombra | El grosor se mide con la transformada de distancia |
| Recorte cuadrado de 256 px | Cuadrado de 256 px centrado en la pieza, con relleno por replicación, reducido a 128×128 | Conserva el tamaño real de la pieza para los descriptores de forma |
| Comparación contra conteo a mano | 60 fotos de entrenamiento contadas a mano, umbral de aceptación de 54 | Elige el método con una medida independiente del clasificador |
| Revisión de contornos grandes | Contornos de más de 1,5 veces la mediana de su clase, revisados a mano | Detecta dos piezas que se tocan; esas fotos salen del banco |
| Mosaicos de control | 24 recortes con su contorno, revisados a ojo | Comprueba el resultado completo, no solo los conteos |

## Fase 3: Características

### Bibliotecas

scikit-image, OpenCV y NumPy; scikit-learn solo para el estudio de HOG.

| Biblioteca | Uso en la fase | Justificación |
|---|---|---|
| `skimage.feature` (scikit-image) | `hog` y `local_binary_pattern` | Implementaciones estándar con los parámetros que pide el enunciado |
| `cv2` (OpenCV) | Gris, contornos, rectángulo mínimo, casco convexo y momentos de Hu | Ya se usa en la fase 2 |
| `numpy` | Vectores, histogramas y archivos `.npz` | `.npz` guarda X, y, nombres y columnas juntos y se carga rápido |
| `sklearn` (scikit-learn) | Estudio de HOG: `StandardScaler`, `SVC` lineal, `accuracy_score` y `f1_score` | Clasificador fijo y simple para comparar configuraciones de HOG en validación |
| `time` | Tiempo de cada configuración del estudio de HOG | Muestra el costo de las configuraciones de mayor dimensión |
| `json` | `caracteristicas\resumen.json` | Deja registradas las dimensiones y los parámetros |

### Técnicas

| Técnica | Uso en la fase | Justificación |
|---|---|---|
| HOG (9 orientaciones, celdas 8×8, bloques 2×2, L2-Hys) | Sobre el gris del recorte de 128×128. Dimensión: 15×15 bloques × 2×2 celdas × 9 = 8.100 | Obligatorio (sección 3.3). Describe los bordes, que distinguen ladrillos de placas |
| Forma sobre el contorno mayor de la máscara | Área, perímetro, proporción (rectángulo mínimo), solidez y 7 momentos de Hu en escala −signo·log10\|hu\| | Obligatorio (rúbrica). El recorte A conserva el tamaño real, así que área y perímetro sirven. El log lleva los Hu a rangos comparables |
| LBP uniforme invariante a rotación, en dos escalas (P=8, R=1 y P=16, R=2) | Un histograma por escala (10 y 18 bins), normalizado, solo con píxeles de la máscara. Las dos escalas las pide el enunciado | Familia adicional de textura. Las piezas aparecen rotadas y de varios colores; LBP no depende del color ni de la rotación y describe los tachones y bordes de la superficie |
| Sin escalado ni PCA | Los vectores se guardan crudos | El escalado y el PCA se ajustan en la fase 4, solo con entrenamiento |
| Estudio de parámetros de HOG | Seis configuraciones; estandarización + SVM lineal (C = 1) entrenada con entrenamiento y medida en validación. El vector final sigue con 9 orientaciones, celdas 8×8 y bloques 2×2 | Lo pide la rúbrica. Muestra cómo cambian la dimensión y el desempeño sin tocar la prueba |
| Guardas de ejecución | Comprueba 951 filas y valores finitos; se detiene si `resumen.json` ya existe | Igual que en las fases 1 y 2 |

## Fase 4: Clasificación

### Bibliotecas

scikit-learn 1.9.1, joblib 1.6.0 y matplotlib.

| Biblioteca | Uso en la fase | Justificación |
|---|---|---|
| `sklearn.pipeline` y `sklearn.compose` | `Pipeline` y `ColumnTransformer`: preprocesamiento + clasificador en un solo objeto | El escalado y el PCA se ajustan dentro del modelo, solo con entrenamiento, y viajan con él a la aplicación |
| `sklearn.preprocessing` | `StandardScaler` por columna | Pone las columnas en la misma escala (el área va en miles y la solidez cerca de 1); las SVM lo necesitan |
| `sklearn.decomposition` | `PCA` al 95 % sobre HOG; PCA completo para la gráfica de varianza; PCA a 2D para visualizar | Reduce las 8.100 columnas de HOG a 494 componentes |
| `sklearn.svm` | `SVC` con núcleo lineal y RBF | Clasificadores que pide el enunciado |
| `sklearn.calibration` | `CalibratedClassifierCV` (sigmoide, 5 partes) sobre la SVM | Da la probabilidad de la SVM; reemplaza a `SVC(probability=True)`, obsoleto en scikit-learn 1.9 |
| `sklearn.ensemble` | `RandomForestClassifier` | Clasificador adicional y medida de importancia de las características |
| `sklearn.metrics` | `accuracy_score` y `f1_score` (macro) | Medidas para elegir en validación |
| `joblib` | Guarda y carga los modelos (`.joblib`) | Formato que pide el enunciado (sección 4) |
| `matplotlib` (backend `Agg`) | Gráficas de varianza, importancia y PCA 2D | Guarda PNG sin abrir ventanas |
| `itertools`, `json`, `shutil`, `time` | Cuadrículas, parámetros en texto, copia del modelo final y tiempos | Biblioteca estándar |

### Técnicas

| Técnica | Uso en la fase | Justificación |
|---|---|---|
| Cinco experimentos | E1 HOG; E2 HOG + forma; E3 LBP; E4 forma + LBP; E5 los tres | Miden qué aporta cada descriptor; cubren E1 a E5 de la rúbrica |
| PCA solo en HOG | Escalado + PCA al 95 % en el bloque HOG; forma y LBP solo escalados | HOG tiene más columnas que piezas; forma y LBP son pocas y se diluirían en un PCA conjunto |
| Búsqueda en cuadrícula con validación | 27 combinaciones por experimento (135 en total), entrenadas con entrenamiento y medidas en validación | Elige hiperparámetros sin tocar la prueba. Sin validación cruzada (opcional en el enunciado) |
| F1 macro para elegir | Con desempates: accuracy, menos columnas, clasificador más simple | Pesa igual a las cuatro clases, que no tienen el mismo número de piezas |
| Importancia con Random Forest | `feature_importances_` de un Random Forest sobre E5 sin PCA, por bloque y por columna | Lo pide el enunciado; sin PCA cada importancia corresponde a una columna original |
| Verificación al final del script | Recarga los 15 modelos, compara cifras, revisa probabilidades y que el escalado vio 674 piezas | Garantiza que lo guardado es lo medido y que no hubo fuga de validación o prueba |

## Fase 5: Evaluación en prueba

### Bibliotecas

Sin bibliotecas nuevas.

| Biblioteca | Uso en la fase | Justificación |
|---|---|---|
| `sklearn.metrics` | `accuracy_score`, `f1_score`, `classification_report`, `ConfusionMatrixDisplay`, `roc_curve`, `auc` y `roc_auc_score` | Métricas que pide el enunciado (3.5) |
| `sklearn.model_selection` | `learning_curve` con `StratifiedGroupKFold` | Curva de aprendizaje con la función que nombra el enunciado, agrupada por bloque |
| `sklearn.preprocessing`, `sklearn.base` | `label_binarize` (ROC uno contra todos) y `clone` (copia sin entrenar del modelo final para la curva) | La curva no toca el modelo guardado |
| `joblib`, `matplotlib`, `cv2` | Carga de modelos, gráficas y lectura de recortes para la galería | Ya usadas en fases anteriores |

### Técnicas

| Técnica | Uso en la fase | Justificación |
|---|---|---|
| Uso único de la prueba | El script se detiene si `tabla_prueba.csv` existe y comprueba que ningún modelo cambió | La prueba no ajusta ni elige nada (`limitaciones.md`, método de evaluación) |
| Tabla de 15 combinaciones | Accuracy y F1 macro en prueba junto a validación | Tabla resumen que pide el enunciado (3.4) |
| Matrices de confusión por experimento | Sin normalizar y normalizadas por clase real, del mejor modelo de cada experimento | Muestra qué confusiones resuelve cada descriptor |
| Galería por confianza | Si el modelo final no tiene errores, muestra sus 8 piezas de menor confianza | El modelo final no falló ninguna pieza en prueba |
| Galería de errores de los otros modelos | `galeria_errores.py` dibuja las piezas de prueba que fallaron los modelos distintos al final, a partir de las predicciones ya guardadas | El modelo final no falló ninguna pieza; así la galería muestra errores reales sin volver a usar la prueba |
| ROC uno contra todos | Con las probabilidades del modelo final | Extra del enunciado, sin costo adicional |
| Curva de aprendizaje agrupada por bloque | 5 grupos por bloque de 10 minutos, 10 tamaños, solo entrenamiento | Diagnóstico de sobreajuste o subajuste sin fuga entre fotos casi iguales |

## Fase 6: Aplicación y robustez

### Bibliotecas

Streamlit 1.66.0.

| Biblioteca | Uso en la fase | Justificación |
|---|---|---|
| `streamlit` | Interfaz: subir la foto, mostrarla anotada, métricas y tablas; `cache_resource` carga el modelo una vez | Una de las opciones del enunciado (3.6); se escribe en Python y se abre en el navegador |
| `cv2`, `numpy` | Rectángulos y etiquetas, composición de escenas, variantes de brillo y fondo | Ya usadas en las fases anteriores |
| `joblib` | Carga de `modelo_final.joblib` | Formato de la fase 4 |
| `random`, `collections`, `time` | Elección de piezas y posiciones (semilla 42), conteos y tiempo por imagen | Biblioteca estándar |

### Técnicas

| Técnica | Uso en la fase | Justificación |
|---|---|---|
| Una sola función de reconocimiento | `reconocer.detectar` la usan la app y la evaluación | Las dos hacen exactamente lo mismo |
| Comprobación contra las fases 3 y 5 | Vector y clase iguales en 5 fotos de prueba | Garantiza que la app reproduce el sistema evaluado |
| Escenas por composición a tamaño real | Piezas de prueba recortadas con su máscara y pegadas sin escalar ni girar sobre el fondo del banco | Sin fotos propias; el tamaño importa porque el modelo usa área y perímetro |
| Variantes de luz y fondo | Brillo × 0,6 y × 1,4, gradiente de × 0,5 a × 1,2 y fondo gris con textura | Prueba de robustez que pide el enunciado |
| Emparejamiento por IoU ≥ 0,5 | Cada pieza real con la caja detectada que más se le superpone | Cuenta aciertos, piezas perdidas y detecciones falsas sin revisar a mano |
