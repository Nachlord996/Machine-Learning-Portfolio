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