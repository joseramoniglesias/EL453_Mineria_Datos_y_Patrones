# Tarea 01: Reconocedor de Letras (VERSION 2022-2)
## Enunciado

El objetivo de esta tarea es diseñar y evaluar un reconocedor automático de las letras X, Y, Z, A, B, en diferentes fonts.

## Descripción

En este [archivo zip](https://github.com/joseramoniglesias/EL453_Ciencia_de_Datos_1/blob/main/Tareas/Tarea_1/letras.zip) se encuentran 100 imágenes de cada una de las cinco letras. Algunas de ellas se muestran a continuación:

![image](https://user-images.githubusercontent.com/38440709/186747306-902ea57d-1a03-4cb0-acb8-5234d24aaa72.png)

El formato del nombre de las imágenes es 'char_nn_kkk.png', donde nn = 01, ... 05, indica el numero de la letra (X,Y,Z,A,B respectivamente), y kkk = 001, 002, ... 100 indica el número de la imagen.

En esta tarea, extraiga características de estas letras de tal forma que se pueda diseñar un clasificador que reconozca de manera automática cuál de las cinco letras es. La idea es que Ud. diseñe el clasificador usando sólo las primeras 75 imágenes de cada letra, y que pruebe el desempeño en las 25 imágenes restantes.

Ud. deberá entregar un Informe y un Código, a continuación, se explica cada uno:

## Especificación del Informe

Ud. deberá entregar un informe de una página (*) en formato pdf en el que explique cómo realizó la extracción de características y la clasificación. Deberá incluir un gráfico del espacio de características y/o histogramas, en los que se visualice la separación de las clases. Además, deberá incluir la matriz de confusión, así como indicar el desempeño del clasificador, es decir qué porcentaje de las letras del conjunto de entrenamiento y del conjunto de pruebas se clasificaron correctamente. Ver más detalles.

(*) Si el informe tiene más de una página solo se revisará la primera página del informe,

## Especificación del código

Ud. deberá entregar también el código de la solución (debidamente ordenado, explicado y comentado) en Python, es decir un archivo ipynb (notebook Jupiter). En el código deberá incluir (y usar) una función llamada 'Reconocedor' que reciba como entrada una única imagen binaria de una letra, y que entregue como salida el número de la letra que ha reconocido. Es decir se pide como salida de la función los números, 1,2,3,4 ó 5, dependiendo si la entrada es X,Y,Z,A,B respectivamente.

* Restricciones:

  * Se puede usar aquellas funciones de librerías de procesamiento de imágenes en las que el input es una imagen, y el output es una imagen.
  * No está permitido usar funciones de librerías que realicen la extracción de características.
  * No está permitido usar funciones de librerías que realicen la clasificación.
  * No está permitido usar funciones de librerías que realicen la evaluación.
  * No hay restricciones en el tiempo de ejecución, sólo esperamos que sea algo razonable para que lo/as ayudantes del curso puedan corregir 150 tareas. Gracias por su comprensión :)

## Fecha de Entrega

17 de septiembre de 2022
