# Limitaciones

Puntos en los que el proyecto no cumple lo que pide el enunciado o en los que sus resultados deben leerse con cuidado.

## Banco de imágenes

- **Sin fotos propias.** Todo el banco viene de Kaggle ("Lego Brick Sorting - Image Recognition").
- **La prueba no incluye otro día ni otra cámara.** Todas las fotos son del mismo montaje y de dos días (8 y 14 de julio de 2018), y los dos días aparecen en entrenamiento. No hay datos de cámara en las fotos, así que no se puede saber si hubo una o varias.
- **Fechas desiguales en prueba.** En prueba, Plate 2x2 y Plate 2x4 solo tienen fotos del 8 de julio, y Brick 2x2 L tiene 47 del 8 y 7 del 14. La partición por bloques no controla la fecha.
- **Prueba pequeña.** La prueba tiene de 1 a 3 bloques de 10 minutos por clase (7 en total). Sus métricas dependen de qué bloques cayeron en ella y podrían variar mucho con otra partición. Una pieza equivale a 0,75 puntos de accuracy en prueba y a 0,7 en validación.
- **Prueba sin casos difíciles.** La prueba no tiene piezas cortadas por el borde y solo tiene 2 fotos con dos piezas, así que sus métricas no dicen nada sobre esos casos.
- **Bloques vecinos.** Dos bloques seguidos pueden mostrar piezas parecidas y caer en conjuntos distintos. No se revisaron a mano.
- **Escenas armadas, no fotografiadas.** Las 10 escenas se componen pegando piezas de prueba sobre el fondo del banco. Las piezas pegadas no tienen la sombra ni el contacto de una foto real, así que las escenas pueden ser algo más fáciles que la realidad.

## Segmentación

- **Piezas claras junto a una oscura.** Con un solo umbral (Otsu) para toda la imagen, una pieza clara puede quedar del lado del fondo y perderse, o quedar absorbida en la máscara de la oscura. Es la principal causa de fallas en las escenas (35 de 190 piezas perdidas).
- **Piezas que se tocan.** Dos piezas pegadas pueden quedar como un solo contorno. Se detectaron y excluyeron las que superan 1,5 veces la mediana de área de su clase; las más pequeñas no se detectan. No se implementó la separación con transformada de distancia y watershed (ya que era opcional).
- **Piezas de más de 256 px.** El recorte es un cuadrado de 256 px; una pieza de validación de 260 px pierde 2 px por lado (No es muy relevante).

## Características y modelo

- **Dependencia del tamaño.** El modelo final se guía sobre todo por el área y el perímetro. Funciona porque la cámara está siempre a la misma distancia; con otra distancia o resolución, la misma pieza parecería de otro tamaño.
- **Piezas cortadas por el borde.** El modelo no se entrenó con ellas. La aplicación las detecta y las marca, pero su clasificación es menos confiable.
- **HOG usa todo el recorte, incluido el fondo.** En el banco el fondo es uniforme; con fondos con textura, el HOG cambiaría. El modelo final no usa HOG, pero los experimentos E1, E2 y E5 sí.
- **Ventaja del modelo final no concluyente.** En validación y en prueba, el modelo final (forma + LBP con SVM lineal) acierta todo, pero varios modelos fallan solo una pieza más o empatan. La diferencia no permite afirmar que sea claramente mejor.

## Método de evaluación

- **El modelo final se eligió con validación, no con prueba.** El enunciado sugiere elegirlo con la tabla de experimentos sobre prueba. Aquí la validación elige los hiperparámetros y el modelo final, y la prueba se usa una sola vez al final para llenar la tabla, sin cambiar nada después. Elegir con la prueba inflaría su resultado (fuga de información). La tabla que pide el enunciado se entrega igual.
- **Sin validación cruzada para elegir.** Es opcional en el enunciado. Se usó un conjunto de validación separado por bloques. La validación cruzada (agrupada por bloque) solo se usó para la curva de aprendizaje.
