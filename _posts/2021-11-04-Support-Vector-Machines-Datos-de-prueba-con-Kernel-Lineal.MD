---
toc: true
layout: post
description: Predecir la variable objetivo binaria a partir de un Dataset reducido usando SVM con Kernel lineal. 
categories: [SVM, Rapidminer]
title: "Support Vector Machines: Predicción de clase para coordenadas en espacio 2D"
---

La clasificación binaria es un problema recurrente en el ámbito de los modelos supervisados de Machine Learning. Al existir exclusivamente dos clases, la visualización de los datos (Mayoritariamente en dos dimensiones) otorga la capacidad de decidir enfrentar con el problema con un tipo de algoritmos en particular.

En este artículo se demostrará el uso de un algoritmo reciente para clasificación binaria llamado "Support Vector Machines". El mismo recoge aspectos de la Estadística, la Teoría Matemática de la Optimización y de las Ciencias de la Computación.

## Contexto

Para este ejercicio se utilizará un Dataset de ejemplo de coordenadas bidimensionales en el plano. Cada una de ellas tendrá una clase correspondiente, la cual será la variable objetivo a predecir del modelo. El ejercicio fue extraído del libro "Predictive Analytics and Data Mining" (Kotu, 2015).

 - Coordenada X1 [Numérica]
 - Coordenada X2 [Numérica]
 - Clase [Binomial (A - B)]
 
El dataset es muy reducido, contiene solamente 17 ejemplos. Por tanto, entrenaremos un modelo de SVM utilizando la totalidad de los datos disponibles. Para testear utilizaremos puntos elegidos estratégicamente que nos permitan valorar su desempeño.

## Datos
#
El dataset a utilizar es el siguiente:
| x1   	| x2   	| Clase 	|
|------	|------	|-------	|
| 1.5  	| 2.5  	| A     	|
| 2    	| 2    	| A     	|
| 1    	| 2    	| A     	|
| 0.75 	| 3    	| A     	|
| 2    	| 1    	| A     	|
| 1.75 	| 1.75 	| A     	|
| 2.75 	| 0.75 	| A     	|
| 2.5  	| 1.5  	| A     	|
| 0.5  	| 6    	| B     	|
| 1.5  	| 6    	| B     	|
| 2    	| 5.5  	| B     	| 
| 1    	| 5.5  	| B     	|
| 1    	| 6.5  	| B     	|
| 2    	| 4.5  	| B     	|
| 1.75 	| 5.25 	| B     	|
| 2.75 	| 4.25 	| B     	|
| 2.5  	| 5    	| B     	|

Los datos a utilizar cumplen con los requisitos de SVM. Todos los predictores son numéricos y la variable objetivo es binaria.

## Modelo

La implemententación del modelo será en Rapidminer. El primer paso será preparar los datos y cargarlos con un operador Retrieve. Es bastante interesante aprovechar esta oportunidad y visualizar los datos mediantes un gráfico plot.

![]({{ site.baseurl }}/images/SVM-Simple-Plot-RM.png "Gráfico de tipo plot (X2 y X1)")

> Con este gráfico se identifica fácilmente la naturaleza de la clasificación a realizar. Ambas clases se encuentran separadas apropiadamente por una "recta imaginaria".
>

Al flujo se debe agregar el operador de _Support Vector Machine_ (SVM) así como también un _Apply Model_. Este último se nutrirá de ejemplos generados manualmente, los cuales cargaremos en un dataset nuevo.

El flujo debería verse de la siguiente manera:

![]({{ site.baseurl }}/images/SVM-Simple-Flow.png "Flujo de modelo SVM en Rapidminer")

## Resultados

El nuevo dataset contiene tres ejemplos (X1, X2) para clasificar: __(1.5, 1), (1.5, 4) y (2, 7)__.

Desde un análisis gráfico, los puntos 1 y 3 son fácilmente ubicables en los sectores donde una única clase predomina, sin embargo el punto 2 está en medio de estas zonas, en una frontera donde la predicción se vuelve más compleja. 

Por esta razón, la confianza de predicción será más alta para los puntos 1 y 3, mientras que el punto 2 tendrá una confianza más equilibrada entre clases.

![]({{ site.baseurl }}/images/SVM-Simple-Resultados-Clasificación.png "Resultados de clasificación de nuevos valores")

>En esta imagen se confirman nuestras suposiciones sobre el comportamiento del modelo.
>

Adicionalmente, vale la pena revisar los pesos generados por el modelo. Volviendo a la teoría, estos representan los coeficientes de cada predictor para partir de hyperespacio usando hyperplanos.

![]({{ site.baseurl }}/images/SVM-Simple-Weights.png "Coeficientes de predictores generados por SVM")