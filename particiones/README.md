# Banco de fotos y partición

Banco de fotos de piezas Lego para la Parte 2 de la Tarea 2 de Procesamiento Digital de Imágenes (Universidad de Antioquia). Describe la fuente, el montaje de las fotos, cómo se repartió el banco en entrenamiento, validación y prueba, y qué contiene cada carpeta. La partición se generó el 11 de octubre de 2026. Las secciones 8 y 9 (agregadas el 16 de octubre) describen el banco final de 951 fotos que usa el modelo y las imágenes de escena.

## 1. Fuente

- Dataset: "Lego Brick Sorting - Image Recognition", usuario pacogarciam3 (Kaggle).
- Cita: https://www.kaggle.com/datasets/pacogarciam3/lego-brick-sorting-image-recognition/data
- El dataset tiene 20 clases. Se usan 4, solo sus fotos base (`Base Images`): 974 fotos. Los recortes (`Cropped Images`) no entran al banco. No hay fotos propias.

## 2. Montaje

Descripción basada en la revisión de 8 fotos (dos por clase, una de cada fecha, elegidas al azar con semilla 42) y en la medición de 64 fotos de trabajo:

- Fondo azul claro, con textura de papel o tela, y sin otros objetos a la vista. En las 64 fotos el tono es azul y el brillo es alto (mediana 230 sobre 255).
- Casi todas las fotos tienen una pieza, ocupando una parte pequeña del encuadre. En la revisión, una de las 8 tenía una segunda pieza cortada por el borde, así que hay fotos con dos piezas (sección 3).
- Vista desde arriba, en ángulo oblicuo: se ven las caras laterales de las piezas.
- Iluminación parecida de una foto a otra: en las 64 fotos, la franja del borde varía 7 niveles entre el percentil 5 y el 95.
- Cámara: ninguna de las 64 fotos revisadas trae datos de cámara (EXIF). No se puede saber si hay una o varias cámaras.

## 3. Fotos por clase y conjunto

| Clase | Entrenamiento | Validación | Prueba | Total | Bloques de prueba |
|---|---|---|---|---|---|
| Brick 1x4 | 169 (71 %) | 40 (17 %) | 29 (12 %) | 238 | 2 |
| Brick 2x2 L | 213 (67 %) | 50 (16 %) | 54 (17 %) | 317 | 3 |
| Plate 2x2 | 153 (74 %) | 31 (15 %) | 23 (11 %) | 207 | 1 |
| Plate 2x4 | 157 (74 %) | 27 (13 %) | 28 (13 %) | 212 | 1 |
| **Total** | **692 (71 %)** | **148 (15 %)** | **134 (14 %)** | **974** | **7** |

Los porcentajes son del total de cada clase. Un bloque es un grupo de fotos de una misma clase tomadas en el mismo intervalo de 10 minutos (sección 5).

Un diagnóstico de calidad automático (`..\limpieza_calidad.py`) marca 10 fotos con dos piezas. En esas fotos, la etiqueta corresponde a la pieza más cercana al centro de la foto.

## 4. Fechas de captura

Según la fecha del nombre de cada archivo (`AAMMDD`, con `AA` = 18, tomado como 2018), las fotos son de dos días: el 8 y el 14 de julio.

| Conjunto | 8 de julio | 14 de julio | Total |
|---|---|---|---|
| Entrenamiento | 241 | 451 | 692 |
| Validación | 89 | 59 | 148 |
| Prueba | 107 | 27 | 134 |

## 5. Regla de partición

1. **Bloque.** Las fotos de una misma clase cuyos nombres (`2_<clase>_AAMMDDHHMMSS.jpg`) caen en el mismo intervalo de 10 minutos forman un bloque. Las fotos seguidas de una misma clase son casi iguales (la mediana de la separación entre ellas va de 13 s a 23 s, según la clase), así que todas las fotos de un bloque van al mismo conjunto. Repartir foto por foto dejaría casi copias en entrenamiento y en prueba.
2. **Reparto por clase.** Cada clase se reparte por separado. Sus bloques se ordenan cronológicamente y se mezclan con `random.Random(42)`, con un generador nuevo por clase.
3. **Asignación.** Cada bloque va al conjunto con mayor déficit frente a su objetivo: 70 %, 15 % y 15 % de las fotos de la clase, medido en fotos. En caso de empate, gana entrenamiento, luego validación y luego prueba.
4. **Comprobación.** En cada clase, cada conjunto queda a ±5 puntos de su objetivo (la mayor desviación es de 4,1 puntos: Plate 2x4 en entrenamiento), y ningún bloque queda repartido entre dos conjuntos.
5. **Semilla.** 42 en todo proceso aleatorio.

## 6. Archivos

- `train.csv` (entrenamiento), `val.csv` (validación) y `test.csv` (prueba): una fila por foto, con las columnas `archivo` (nombre del archivo, sin ruta), `clase` (`Brick_1x4`, `Brick_2x2_L`, `Plate_2x2` o `Plate_2x4`) y `bloque` (`AAMMDDHHMM`).
- `train\`, `val\` y `test\`: copia de las fotos, una carpeta por clase. Son para la entrega; los CSV son la partición de referencia.
- `..\Codigo\particion.py`: genera los CSV y las copias, con solo la biblioteca estándar de Python. Se ejecuta desde `Taller 2` con `python Codigo\particion.py`. Si `particiones\` ya existe, no escribe nada.

## 7. Limitaciones

- La prueba tiene de 1 a 3 bloques por clase (7 en total). Sus métricas dependen de qué bloques caigan en ella y pueden variar mucho.
- La prueba no incluye fotos de otro día (sección 3.1 del enunciado). Sus fotos son de las mismas dos fechas que el entrenamiento. Sobre otra cámara no hay forma de comprobarlo (sección 2).
- La distribución por fecha no es uniforme entre clases. En prueba, Plate 2x2 y Plate 2x4 solo tienen fotos del 8 de julio, y Brick 2x2 L tiene 47 del 8 de julio y 7 del 14. La regla de bloques no controla la fecha. La partición es final y no se regenera.
- Bloques vecinos pueden mostrar piezas parecidas y caer en conjuntos distintos. No se revisaron a mano; cualquier fuga residual entre bloques vecinos queda como limitación.
- Según el diagnóstico de calidad, la prueba no tiene piezas cortadas por el borde y tiene dos fotos con dos piezas, ambas del mismo bloque (Plate 2x4). Por eso sus métricas no cubren piezas de borde y dicen muy poco sobre fotos con dos piezas.

## 8. Banco final usado por el modelo (fase 2)

La partición de las secciones 3 a 5 es la de las 974 fotos. Al segmentar las piezas (fase 2), el banco que usan las características, el entrenamiento y la evaluación quedó en **951 fotos, una pieza por foto**. Las reglas se resumen abajo.

| Clase | Entrenamiento | Validación | Prueba | Total |
|---|---|---|---|---|
| Brick 1x4 | 165 | 38 | 29 | 232 |
| Brick 2x2 L | 213 | 50 | 54 | 317 |
| Plate 2x2 | 151 | 30 | 23 | 204 |
| Plate 2x4 | 145 | 26 | 27 | 198 |
| **Total** | **674** | **144** | **133** | **951** |

Diferencias con la partición original (974 fotos):

- **10 fotos excluidas por piezas dobles** (5 de entrenamiento, 4 de validación y 1 de prueba): dos piezas que se tocan quedaban como un solo contorno bajo una sola etiqueta. Lista en `..\resultados\fase2\fotos_excluidas.csv`.
- **13 fotos de entrenamiento sin pieza utilizable:** 11 con dos o más piezas (el entrenamiento no usa fotos de varias piezas) y 2 cuya pieza toca el borde de la foto.
- **Validación y prueba** conservan las fotos de dos piezas y las piezas de borde; en las fotos de dos piezas solo cuenta la más cercana al centro.
- Las piezas que no entran a ningún conjunto se listan con su motivo en `..\resultados\fase2\piezas_no_usadas.csv`.

Los recortes de 128×128 de cada pieza, con su máscara, están en `..\roi\` (`imagenes\`, `mascaras\` y `metadatos.csv`, con la columna `uso` = `train`, `val` o `test`). Las carpetas `train\`, `val\` y `test\` de esta partición conservan las 974 fotos originales.

## 9. Imágenes de escena

El enunciado pide al menos 10 fotos con varias piezas a la vez. No se tomaron fotos propias, así que las escenas se **armaron por composición** (fase 6, `..\Codigo\generar_escenas.py`):

- **10 escenas** en `..\escenas\` (`escena_01.png` a `escena_10.png`), con 3 a 5 piezas cada una (38 en total).
- Cada pieza se tomó de una foto de **prueba**, se recortó con su máscara y se pegó **a tamaño real**, sin girar ni escalar, sobre el fondo del propio banco (`background_backlit_A.jpg`), sin solaparse. Semilla 42.
- Cada escena tiene 4 variantes para la prueba de robustez: oscura, clara, luz desigual y otro fondo (`escena_NN_<variante>.png`).
- La clase y la caja de cada pieza pegada están en `..\escenas\verdad.csv`.

Limitación: las piezas pegadas no tienen la sombra que tendría una foto real con varias piezas.
