# Metodología

Clasificación de piezas Lego con características diseñadas a mano. Parte 2 de la Tarea 2 de Procesamiento Digital de Imágenes (Universidad de Antioquia, 2026-2).

Este documento resume, fase por fase, qué se hizo, con qué parámetros y por qué. Las bibliotecas y técnicas de cada fase están en `bibliotecas_y_tecnicas.md` y las limitaciones en `limitaciones.md`. Las rutas son relativas a la raíz del repositorio.

---

## 1. Banco de imágenes y partición

**Fuente.** "Lego Brick Sorting - Image Recognition" (pacogarciam3, Kaggle): https://www.kaggle.com/datasets/pacogarciam3/lego-brick-sorting-image-recognition/data. El banco tiene 20 clases; se usan 4 y solo sus fotos base (`Base Images`), no los recortes que trae el banco.

| Clase       | Fotos   | 8 de julio de 2018 | 14 de julio de 2018 |
| ----------- | ------- | ------------------ | ------------------- |
| Brick 1x4   | 238     | 125                | 113                 |
| Brick 2x2 L | 317     | 110                | 207                 |
| Plate 2x2   | 207     | 105                | 102                 |
| Plate 2x4   | 212     | 97                 | 115                 |
| **Total**   | **974** | **437**            | **537**             |

Montaje: fondo azul claro y uniforme, una pieza por foto en la mayoría, vista oblicua desde arriba, iluminación muy pareja entre fotos y piezas de varios colores en cada clase. Todas las fotos miden 640×480.

**Partición 70/15/15 por bloques de 10 minutos** (`Codigo/particion.py`).

- Las fotos seguidas de una misma clase están separadas entre 13 y 23 s (mediana por clase) y son casi iguales. Repartirlas foto por foto dejaría casi copias en entrenamiento y en prueba (fuga de información).
- Por eso las fotos de una clase cuyo nombre (`2_<clase>_AAMMDDHHMMSS.jpg`) cae en el mismo intervalo de 10 minutos forman un **bloque**, y cada bloque va entero a un conjunto.
- Cada clase se reparte por separado (estratificación). Sus bloques se ordenan cronológicamente, se mezclan con semilla 42 y cada uno va al conjunto con mayor déficit frente a 70/15/15, medido en fotos.
- Resultado: entrenamiento 692 (71 %), validación 148 (15 %), prueba 134 (14 %). Desviación máxima frente al objetivo: 4,1 puntos (tolerancia ±5).

**Uso de cada conjunto.** Entrenamiento ajusta los modelos; validación elige hiperparámetros y el modelo final; prueba se usa **una sola vez**, al final, y no elige ni ajusta nada.

---

## 2. Preprocesamiento y segmentación

Cada foto pasa por estos pasos (`Codigo/preprocesamiento.py`):

1. **Normalización:** lado mayor a 640 px con `INTER_AREA`.
2. **Filtro:** mediana 3×3 sobre la imagen en color.
3. **Segmentación:** el gris se divide por una estimación del fondo y se umbraliza con Otsu.
4. **Morfología:** apertura y cierre con núcleo elíptico de 5×5.
5. **Contornos:** externos, con área de 150 px o más y un grosor mínimo de 9 px.
6. **Recorte (ROI):** cuadrado de 256 px centrado en la pieza, reducido a 128×128, con su máscara.

### Filtro: mediana 3×3

- En el notebook de la Parte 1 (10 casos: 2 imágenes × 5 tipos de ruido) es la mejor configuración fija: SSIM medio 0,779 y PSNR medio 28,5 dB. Con la mejor configuración de cada familia, la mediana también tiene el mejor SSIM (0,835, frente a 0,782 del gaussiano y 0,754 del bilateral).
- Casi no altera la segmentación: en 30 fotos de entrenamiento no cambió el número de piezas en ninguna y el borde se desplazó 0,05 px de mediana. El gaussiano con σ = 1 cambió el conteo en 2 fotos y desplazó el borde 0,30 px.
- Las fotos tienen ruido gaussiano muy bajo (σ ≈ 0,5 niveles de gris), así que el suavizado aporta poco; se eligió el filtro que menos altera los bordes.

### Corrección de luz: no se aplica

Se fijó una regla antes de medir: aplicar CLAHE solo si el brillo de una franja de 10 px en el borde varía 40 niveles o más entre las fotos de entrenamiento (percentiles 5 a 95). Varía 6 niveles (de 183 a 189), así que no se corrige.

### Segmentación: gris dividido por el fondo + Otsu

- **Fondo:** mediana de 63 px sobre la foto reducida 4 veces (equivale a 252 px en la foto original, más que la pieza más grande). Dividir el gris por ese fondo deja el fondo cerca de 255 en toda la foto y quita las variaciones de luz dentro de ella.
- **Polaridad:** de los dos lados del umbral de Otsu se elige el que deja menos píxeles en las cuatro esquinas de 20×20 px; las esquinas casi siempre son fondo.
- **Elección del método:** se compararon tres variantes contra un conteo a mano de piezas en 60 fotos de entrenamiento (15 por clase, semilla 42), con un umbral de aceptación fijado antes de medir (54 de 60):

| Método                                    | Fotos con el número de piezas correcto |
| ----------------------------------------- | -------------------------------------- |
| Otsu sobre saturación HSV                 | 53 de 60                               |
| Otsu sobre gris                           | 48 de 60                               |
| **Otsu sobre gris dividido por el fondo** | **60 de 60**                           |

### Recorte: cuadrado de 256 px reducido a 128×128

El recorte conserva el tamaño real de la pieza, así que el área y el perímetro son comparables entre fotos. Estirar cada pieza hasta llenar el recorte borraría esa diferencia. Solo una pieza usada supera los 256 px (una Plate 2x4 de validación de 260 px de alto) y pierde 2 px por lado.

### Qué piezas entran a cada conjunto

- **Fotos con dos o más piezas:** no entran al entrenamiento. En validación y prueba se usa la pieza más cercana al centro de la foto, que es la que corresponde a la etiqueta.
- **Piezas que tocan el borde de la foto:** no entran al entrenamiento; en validación y prueba se conservan.
- **Dos piezas pegadas en un solo contorno:** se revisaron a mano los contornos de más de 1,5 veces la mediana de área de su clase. Las 10 fotos con dos piezas pegadas (5 de entrenamiento, 4 de validación y 1 de prueba) salen del banco.

**Banco final: 951 fotos, una pieza por foto.**

| Clase       | Entrenamiento | Validación | Prueba  |
| ----------- | ------------- | ---------- | ------- |
| Brick 1x4   | 165           | 38         | 29      |
| Brick 2x2 L | 213           | 50         | 54      |
| Plate 2x2   | 151           | 30         | 23      |
| Plate 2x4   | 145           | 26         | 27      |
| **Total**   | **674**       | **144**    | **133** |

---

## 3. Características

Cada recorte se convierte en un vector de **8.139 valores** (`Codigo/caracteristicas.py`). No se escala ni se reduce aquí; eso se ajusta en la fase de clasificación, solo con entrenamiento.

| Bloque      | Parámetros                                                                                                                                                                                                                 | Dimensión |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| HOG         | 9 orientaciones, celdas de 8×8, bloques de 2×2, normalización L2-Hys, sobre el gris del recorte. 128/8 = 16 celdas por lado → 15×15 bloques × 4 celdas × 9 orientaciones                                                   | 8.100     |
| Forma       | Sobre el contorno mayor de la máscara: área, perímetro, proporción (lado mayor / lado menor del rectángulo mínimo), solidez (área / área del casco convexo) y los 7 momentos de Hu en escala −signo·log10\|hu\|            | 11        |
| Textura LBP | Patrones binarios locales uniformes e invariantes a rotación en dos escalas: P = 8, R = 1 (10 bins) y P = 16, R = 2 (18 bins). Un histograma por escala, contado solo con los píxeles de la máscara y normalizado a suma 1 | 28        |

**Por qué estos descriptores.**

- **HOG** describe la dirección de los bordes: tachones, aristas y la forma en L.
- **Forma:** el área y la proporción separan las clases por tamaño y alargamiento; los momentos de Hu son invariantes a rotación y escala. El logaritmo lleva los Hu (de 10⁻¹ a 10⁻¹⁷) a rangos comparables.
- **LBP** describe la textura local sin depender del color ni de la rotación. Se probó primero un descriptor de color (media y desviación de H, S y V dentro de la máscara) y se descartó: las cuatro clases tienen piezas de varios colores y las medias por clase casi no cambiaban (V entre 91 y 97; S entre 120 y 149).

**Medias de forma por clase (entrenamiento, escala del recorte de 128).**

| Clase       | Área  | Perímetro | Proporción | Solidez |
| ----------- | ----- | --------- | ---------- | ------- |
| Brick 1x4   | 4.246 | 310       | 2,52       | 0,95    |
| Brick 2x2 L | 2.807 | 230       | 1,19       | 0,92    |
| Plate 2x2   | 2.446 | 205       | 1,17       | 0,97    |
| Plate 2x4   | 5.171 | 315       | 1,82       | 0,97    |

El par más difícil por forma es Brick 2x2 L contra Plate 2x2 (proporción casi igual y áreas cercanas); los separa sobre todo la solidez, por la muesca de la L.

**Estudio de parámetros de HOG** (`Codigo/estudio_hog.py`). Seis configuraciones con un clasificador fijo (estandarización + SVM lineal, C = 1), entrenado con entrenamiento y medido en validación (144 piezas):

| Orientaciones | Celda   | Bloque  | Dimensión | Accuracy en validación | Segundos |
| ------------- | ------- | ------- | --------- | ---------------------- | -------- |
| **9**         | **8×8** | **2×2** | **8.100** | **0,993**              | **13,2** |
| 6             | 8×8     | 2×2     | 5.400     | 0,993                  | 10,7     |
| 12            | 8×8     | 2×2     | 10.800    | 0,993                  | 81,0     |
| 9             | 16×16   | 2×2     | 1.764     | 0,993                  | 4,3      |
| 9             | 4×4     | 2×2     | 34.596    | 0,986                  | 155,2    |
| 9             | 8×8     | 3×3     | 15.876    | 0,993                  | 64,5     |

Los parámetros casi no cambian el resultado (cinco configuraciones fallan una pieza de 144); sí cambian el costo. Celdas de 16×16 dan lo mismo con 4,6 veces menos valores. Pero se mantiene la configuración que se eligio al inicio.

---

## 4. Clasificación

`Codigo/entrenar.py`. Todos los modelos se entrenan **solo con entrenamiento** (674 piezas) y se miden en validación (144). La prueba no se usa en esta fase.

**Cinco experimentos:**

| Experimento | Descriptores      | Valores |
| ----------- | ----------------- | ------- |
| E1          | HOG               | 8.100   |
| E2          | HOG + forma       | 8.111   |
| E3          | LBP               | 28      |
| E4          | Forma + LBP       | 39      |
| E5          | HOG + forma + LBP | 8.139   |

E3 usa LBP solo para ver qué logra esa familia sin HOG; lo que aporta LBP sobre HOG se ve comparando E2 con E5.

**Preprocesamiento dentro de cada modelo.** Estandarización por columna y, solo para el bloque HOG, PCA con los componentes que explican el 95 % de la varianza (494 componentes). Forma y LBP no pasan por PCA: son pocas columnas y un PCA conjunto las diluiría entre las 8.100 de HOG. El escalado y el PCA se ajustan solo con entrenamiento y quedan guardados dentro de cada modelo.

**Clasificadores y búsqueda de hiperparámetros** (27 combinaciones por experimento, 135 en total):

| Clasificador  | Valores probados                                                 |
| ------------- | ---------------------------------------------------------------- |
| SVM lineal    | C ∈ {0,01; 0,1; 1; 10; 100}                                      |
| SVM RBF       | C ∈ {0,1; 1; 10; 100} × γ ∈ {0,0001; 0,001; 0,01; `scale`}       |
| Random Forest | árboles ∈ {200; 500} × profundidad máxima ∈ {10; 20; sin límite} |

- Cada combinación se entrena con entrenamiento y se mide en validación; no se usa validación cruzada (ya hay un conjunto de validación separado por bloques).
- **Criterio:** F1 macro en validación, porque pesa igual a las cuatro clases y estas no tienen el mismo número de piezas. Desempates: accuracy, después el modelo con menos columnas y después el clasificador más simple (SVM lineal, SVM RBF, Random Forest).
- **Probabilidades:** las SVM se calibran con el método sigmoide (Platt) en 5 partes del entrenamiento; el Random Forest usa sus propias probabilidades.

**Resultados en validación** (F1 macro; entre paréntesis, piezas falladas de 144):

| Experimento    | SVM lineal    | SVM RBF    | Random Forest |
| -------------- | ------------- | ---------- | ------------- |
| E1 HOG         | 0,993 (1)     | 0,986 (2)  | 0,978 (3)     |
| E2 HOG + forma | 0,993 (1)     | 0,986 (2)  | 0,993 (1)     |
| E3 LBP         | 0,789 (28)    | 0,800 (26) | 0,805 (26)    |
| E4 forma + LBP | **1,000 (0)** | 1,000 (0)  | 0,993 (1)     |
| E5 los tres    | 0,993 (1)     | 0,979 (3)  | 0,993 (1)     |

**Modelo final: E4 (forma + LBP) con SVM lineal, C = 0,1.** Empató con E4 SVM RBF y ganó por ser el clasificador más simple. Usa 39 valores por pieza y no necesita PCA. Entre fallar 0 y fallar 1 pieza hay una sola pieza de diferencia, así que la ventaja en validación no es concluyente.

**Importancia de características** (Random Forest sobre E5 sin PCA, para que cada importancia corresponda a una columna original): HOG suma 0,865 del total, forma 0,117 y LBP 0,018. Por columna, la forma pesa unas 100 veces más que HOG: las 8 columnas más importantes son de forma (perímetro, área, proporción, hu2, hu1, solidez, hu4 y hu3).

---

## 5. Evaluación en prueba

`Codigo/evaluar.py`, ejecutado una sola vez. Después de verlo no se cambió ningún modelo.

**Los 15 modelos en prueba** (F1 macro; entre paréntesis, piezas falladas de 133):

| Experimento    | SVM lineal    | SVM RBF    | Random Forest |
| -------------- | ------------- | ---------- | ------------- |
| E1 HOG         | 1,000 (0)     | 0,961 (5)  | 0,931 (8)     |
| E2 HOG + forma | 1,000 (0)     | 0,961 (5)  | 0,993 (1)     |
| E3 LBP         | 0,783 (28)    | 0,866 (18) | 0,835 (20)    |
| E4 forma + LBP | **1,000 (0)** | 1,000 (0)  | 0,993 (1)     |
| E5 los tres    | 1,000 (0)     | 0,961 (5)  | 1,000 (0)     |

- **Modelo final:** 133 de 133. Precisión, recall y F1 de 1,0 en las cuatro clases; AUC de 1,0. Su pieza de menor confianza tiene probabilidad 0,93. El resultado coincide con el de validación.
- **SVM lineal** no falla en prueba con ningún experimento que incluya HOG o forma.
- **SVM RBF con HOG** (E1, E2 y E5) falla las mismas 5 piezas con probabilidades muy altas (0,93 a 0,9999).
- **E3 (LBP solo)** acierta entre el 79 % y el 86 %; su confusión principal es Brick 1x4 tomado como Brick 2x2 L y Plate 2x2 tomado como Plate 2x4.

**Galería de errores.** Como el modelo final no falló ninguna pieza, la galería (`Codigo/galeria_errores.py`) muestra las 9 piezas de prueba que fallaron los demás modelos (sin E3), con 25 errores en total. Causas probables:

- Un Brick 1x4 visto de lado y en escorzo, que parece corto y con un saliente como una L; lo fallan 6 modelos, siempre como Brick 2x2 L.
- Piezas vistas por debajo (tubos en lugar de tachones), poco frecuentes en el banco; la SVM RBF las toma por Plate 2x2 con probabilidades de 0,98 a 1,00.
- Plate 2x4 de un mismo bloque de prueba que el Random Forest con solo HOG confunde con Brick 1x4 (la otra pieza alargada), con poca seguridad (0,34 a 0,43).

**Curva de aprendizaje** (modelo final, `learning_curve` con validación cruzada de 5 grupos armados por bloque de 10 minutos, solo sobre entrenamiento): con 53 piezas llega a F1 0,95, desde unas 160 se estabiliza en 0,99, y la curva de entrenamiento queda entre 0,996 y 0,999. La distancia final entre las dos es de 0,005: no hay sobreajuste ni subajuste, y más datos casi no mejorarían el resultado.

---

## 6. Aplicación y prueba de robustez

**Aplicación** (`Codigo/app.py`, Streamlit). Abre una foto, repite con ella los pasos de las secciones 2 y 3 (`Codigo/reconocer.py`) y aplica el modelo final guardado. Muestra un rectángulo por pieza con su clase y confianza, el conteo por clase y el tiempo de procesamiento. No entrena nada. Se comprobó que en 5 fotos de prueba calcula exactamente el mismo vector de 8.139 valores y la misma clase que la evaluación.

**Escenas** (`Codigo/generar_escenas.py`). Sin fotos propias, se armaron 10 escenas por composición: piezas de fotos de **prueba** recortadas con su máscara y pegadas a tamaño real (sin girar ni escalar, porque el modelo usa área y perímetro) sobre el fondo del propio banco, sin solaparse. Tienen de 3 a 5 piezas (38 en total). Cada escena tiene 4 variantes: oscura (brillo × 0,6), clara (× 1,4), luz desigual (gradiente de × 0,5 a × 1,2) y otro fondo (gris con textura).

**Resultados** (`Codigo/evaluar_escenas.py`). Una pieza está detectada si una caja de la aplicación la cubre con IoU ≥ 0,5:

| Tipo         | Piezas  | Bien clasificadas | Mal clasificadas | Perdidas | Detecciones falsas | % bien   |
| ------------ | ------- | ----------------- | ---------------- | -------- | ------------------ | -------- |
| Original     | 38      | 33                | 0                | 5        | 1                  | 86,8     |
| Oscura       | 38      | 32                | 1                | 5        | 1                  | 84,2     |
| Clara        | 38      | 29                | 1                | 8        | 2                  | 76,3     |
| Luz desigual | 38      | 30                | 0                | 8        | 3                  | 78,9     |
| Otro fondo   | 38      | 29                | 0                | 9        | 0                  | 76,3     |
| **Total**    | **190** | **153**           | **2**            | **35**   | **7**              | **80,5** |

- **El clasificador casi no falla:** de las 155 piezas detectadas, 153 quedan bien clasificadas (98,7 %).
- **Las fallas son de segmentación:** las piezas perdidas son casi todas claras (25 Plate 2x2 beige y 7 Brick 2x2 L amarillas o naranjas). Con un solo umbral para toda la imagen, una pieza clara junto a una oscura queda del lado del fondo. Las detecciones falsas son pedazos de piezas claras segmentadas a medias.
- **Oscurecer casi no afecta;** aclarar, la luz desigual y el otro fondo bajan los aciertos al 76–79 %.
- **Tiempo:** 58 a 74 ms por imagen.
