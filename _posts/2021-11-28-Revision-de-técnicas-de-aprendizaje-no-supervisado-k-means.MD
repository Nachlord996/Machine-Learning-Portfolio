---
toc: true
layout: post
description: "Análisis de grupos o clústers en pacientes con enfermedad coronaria"
categories: ["Clustering", "K-means", "Rapidminer"]
title: "Revisión de técnincas de aprendizaje no supervisado: K-means"
---

Los métodos de aprendizaje automático se pueden dividir en dos grandes enfoques, los supervisados y los no supervisados. El primero se basa en el uso de Datasets etiquetados. Esta información es utilizado en el entrenamiento de un modelo para que este aprenda las relaciones que existen entre los datos de entrada y una variable objetivo. Por esto mismo, este enfoque se centra en generar la capacidad de predicción a partir de ejemplos nuevos, con una precisión aceptable.

Por otra parte, los algoritmos de Machine Learning no supervisados tienen el objetivo de trabajar con datos no etiquetados. No existe una variable objetivo a predecir ni un comportamiento en particular para simular. Este tipo de algoritmos se caracterizan por encontrar patrones y otorgar mayor información sobre un conjunto de datos existente. Dentro de este conjunto se encuentra el algoritmo a tratar en este artículo: K-Means.

K-Means es un algoritmo no supervisado de Clustering. Está pensado para descubrir los distintos grupos que puedan existir en un Dataset sin ese conocimiento previamente. 

## Contexto

En este ejercicio se cuenta una base de datos de pacientes de un hospital. Se desea obtener más información acerca de la influencia del peso, el sexo y el colesterol en el desarrollo de una enfermedad coronaria.

Esta tarea implica detectar la existencia de grupos naturales en los datos que relacionen estos factores mencionados con el desarrollo de la enfermedad. El modelo se entrenará utilizando la herramiento Rapidminer.

## Datos

Se cuenta con un dataset con 3 variables, dos de estas numércias y una binomial.

- Sexo
- Peso
- Colesterol

El dataset tiene 547 registros y no hay valores faltantes. Los datos parecen ser correctos y estar expresados en las unidades adecuadas.

![]({{ site.baseurl }}images\K-means-dataset-plot.png "Gráfico plot del Dataset Cardio")

## Modelo

En el flujo de Rapidminer se debe agregar un operador _K-means_ que entrenará el modelo de clustering. El mismo necesita un parámetro _k_ que significa la cantidad de clusters a generar. Por conocimiento previo del Dataset, se ingresará el valor K = 4, aunque esto normalmente se decide en base a la exploración realizada del Dataset.

Además se puede insertar un operador de visualización de clústeres para poder entender mejor los resultados generados por el modelo. El entrenamiento determina una lista de centroides, puntos en el hyperespacio que representan los centros de los grupos generados. Los demás puntos determinan el grupo al cual pertenecen por medio de la distancia a los centroides.

El flujo se ve de la siguiente manera:

![]({{ site.baseurl }}images\K-means-rapidminer-flow.png "Flujo de Rapidminer de clustering")

La salida que genera el último operador aporta un gráfico de Scatter Plot que muestra visualmente los grupos generados. La utilidad de este modelo podría ser identificar personas que pertenezcan a los clústers de mayor peso y colesterol, ya que estos son los más propensos a desarrollar una enfermedad cardíaca.

![]({{ site.baseurl }}images\K-means-results-plot.png "Vista scatter plot de los clústers resultantes")
