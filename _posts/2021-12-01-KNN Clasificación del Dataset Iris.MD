---
toc: true
layout: post
description: Se analiza el Dataset Iris de UCI para aplicar K Nearest Neighbours.
categories: [K Nearest Neighbours, Rapidminer, No Lineal]
title: "Un problema clásico de clasificación: UCI Iris mediante K-NN"
---

Dentro de las técnicas para clasificación en problemas no lineales, existe un grupo que utiliza la distancia entre predictores como elemento central. Esto parte de la idea de que los ejemplos que pertenecen a una misma clase, serán ciertamente más similares en cuanto a valores de los distintos predictores.

_K-nearest Neighbours_ o simplemente __K-NN__, es un algoritmo para problemas de clasificación que se basa en almacenar la información del dataset y evaluar nuevos ejemplos a partir de un cálculo simple de distancia. La clase predicha de un elemento estará dada por los K elementos (previamente clasificados) más cercanos.

## Contexto

Para este ejercicio se utilizará como base el Dataset Iris de UCI, el mismo es de acceso totalmente público y gratuito. En Rapidminer generaré un flujo que entrene un modelo a partir de los datos de Iris, en este caso bajo la proporción _70/30_, para luego testearlo, revisar los resultados y las distintas opciones de configuración que el modelo ofrece.

## Datos

El Dataset Iris es bastante reducido, cuenta solamente con 150 ejemplos. Además de la variable objetivo, existen 4 predictores distintos relacionados con medidas de las flores observadas en la realidad.

El dataset está completamente etiquetado y ningún predictor tiene valores faltantes. La siguiente tabla resume las variables disponibles en el Dataset:

| Variable         	| Tipo de dato 	|
|------------------	|--------------	|
| Largo Sépalo     	| Numérico     	|
| Ancho del Sépalo 	| Numérico     	|
| Largo del Pétalo 	| Numérico     	|
| Ancho del Pétalo 	| Numérico     	|

La variables objetivo es del tipo Polinomial y tiene __3 valores__ posibles: Iris Setosa, Iris Versi-color e Iris Virginica.

Examinando los gráficos de tipo plot entre los predictores, resulta interesante la agrupación generada en el espacio de las clases para las variables __"Largo Pétalo"__ y __"Ancho Pétalo"__. Además, estos dos predictores son los que tienen mayor correlación con la variable objetivo.

![]({{ site.baseurl }}/images/KNN3.png "Plot de Largo Pétalo y Ancho Pétalo")

## Modelo

En Rapidminer, se importa el Dataset de Iris. En primer lugar, es necesario eliminar los predictores que no vamos a utilizar para el ejercicio. Luego, se realiza la separación de los datos mediante muestreo aleatorio en 70% para Entrenamiento y 30% para Test, esto se puede realizar mediante el operador _"Split Data"_.

Agregamos el operador de K-NN con la configuración que viene por defecto, seguido de un _"Apply Model"_ para predecir los valores de la clase objetivo para el 30% reservado para Test.

El flujo debería verse así:

![]({{ site.baseurl }}/images/KNN1.png "Flujo de Rapidminer para apicar K-NN al Dataset Iris")

## Evaluación

Al ejecutar el modelo, vemos que los resultados obtenidos son muy positivos. Esto quiere decir que nuestro modelo aprovechó bien la correlación entre predictores y la separación en el hyperespacio de las clases objetivo.

Aquí la matriz de confusión para esta ejecución inicial:
![]({{ site.baseurl }}/images/KNN2.png "Matriz de confusión")

Vale destacar que el operador de K-NN ofrece distintos métodos para el cálculo de la distincia, los cuales podrían ser interesantes para otras situaciones donde la separación no sea tan clara o el resultado obtenido no sea tan bueno.


 - Distancia de Chebychev
 - Distancia Euclideana
 - Similitud Coseno
 - Distancia de Manhattan

También, Rapidminer ofrece la posibilidad de ponderar el voto en la selección de la clase objetivo. Esto se puede establecer mediante un _switch_ el cual determina si la distancia importa en la decisión. Es decir, aquellos vecinos cercanos tendrán una mayor incidencia en la clase escogida.
