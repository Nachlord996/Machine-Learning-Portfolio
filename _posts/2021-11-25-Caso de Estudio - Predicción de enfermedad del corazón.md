---
toc: true
layout: post
description: ""
categories: ["Caso de estudio", "Árbol de decisión", "Random Forest", "Rapidminer", "Preparación de los datos", "Entrenamiento"]
title: "Caso de Estudio: Detección de enfermedad cardíaca"
sticky_rank: 1
---
## Contexto del problema

Me resulta particularmente interesante las posibilidades de aplicación de la Ciencia de Datos en el área de la Medicina. Los Hospitales y centros médicos en general, son instituciones que generan y registran información sobre el estado de sus pacientes diariamente. Por lo que, en principio es un buen campo de investigación para proyectos de esta índole.

La información puede ser aprovechada mediante técnicas de Machine Learning para el entrenamiento de modelos que aporten valor a la labor médica. Sin embargo, la mayoría de las instituciones no estarán dispuestas a compartir dicha información para proteger así la privacidad de sus usuarios. En definitiva, los pacientes depositan su confianza en estos centros médicos y tienen el derecho legítimo de que su información personal no sea filtrada. Este derecho se ve respaldado por la obligación, del otro lado del mostrador, de resguardar esta información. Esto último deriva en que la disponibilidad de información sea mayormente escasa y difícil de conseguir.

En este caso de estudio se pretende analizar información de pacientes para la detección de enfermedades cardíacas. Se parte de Datasets que están disponibles públicamente y han sido objeto de estudio en varios artículos sobre esta temática. Estos datos provienen de 4 bases de datos distintas de:

 1. Cleveland Clinic Foundation
 1. Hungarian Institute of Cardiology, Budapest
 1. V.A. Medical Center, Long Beach, CA
 1. University Hospital, Zurich, Switzerland  

El objetivo principal es demostrar las habilidades aprendidas considerando todas las etapas de un proyecto de Machine Learning.

## Preparación de los Datos

### Primera aproximación

A pesar de que los Datasets tienen orígenes distintos, los atributos que los componen se mantienen en todos los Datasets. Por lo tanto, podemos afirmar que una posible unificación de los datos será correcta y beneficiosa para el estudio, ya que contaremos con mayor cantidad de ejemplos y todos tienen la misma estructura. Esto implica también, que los datos registrados estén en las _mismas unidades_ y _encodeados_ de la misma manera. Diversos tipos de datos pueden tener el mismo significado pero tener distinta representación. Durante la preparación de los datos, debemos prestar especial atención a este aspecto ya que es uno de los errores más frecuentes y es fácil pasar por alto.

Como primera aproximación, deberemos investigar la estructura de los archivos, para luego poder levantarlos correctamente en la herramienta escogida.

> Cada uno de los extractos de las bases de datos estan en archivos individuales __.DATA__. Existe además un archivo __.NAMES__ que detalla la información presente en los anteriormente mencionados.
> 

![]({{ site.baseurl }}/images/Validacion-preparacion-de-datos.png "Contenido del archivo Cleveland.data")

La estructura de los datos no parece ser la más conveniente para su lectura, por lo que surge la necesidad de realizar transformaciones de formato. Para ello, realizaremos un script que convierta cada Dataset a un archivo __.csv__ independiente.

### Reestructuración de archivos de datos

Analizando la estructura de este archivo encuentro un patrón, cada cierta cantidad de líneas (En principio 10) hay una que finaliza la palabra "_name_". De acuerdo con el archivo "_.Names_" asociado a estos Datasets, esta variable representa el último atributo del Dataset, por lo que podría ser utilizado para delimitar registros. En el caso de todos los valores intermedios, cada uno está asociado a un predictor según el orden descripto en el archivo "_.Names_".

Vale destacar también, que el __primer valor__ de cada registro se ve incrementando secuencialmente y es distinto para cada uno de los grupos, por lo que coincide con la descripción de __Identificador__ que se esperaba. Los identificadores son un tipo de datos que no debe ser incluido en el proceso de entrenamiento de un modelo. El mismo no aporta información relevante para el problema que se está analizando. La utilidad que aporta este dato es exclusivamente referencial, para que el lector pueda identificar registros en particular.

El script fue escrito en Javascript y ejecutado en un entorno local de NodeJS. A continuación se presenta el código utilizado para estandarizar la información en un formato legible y universal CSV. El script fue ejecutado para cada uno de los archivos del caso de estudio.
Simplemente se leen los archivos en busca de la palabra clave "_name_", mientras que se van acumulando los valores obtenidos hasta el momento. Al encontrar esta palabra, se verifica la integridad del registro encontrado para decidir si será considerado o no. 
Por último se escriben las líneas en el archivo _.csv_ de salida. 

![]({{ site.baseurl }}/images/Script-Limpieza-de-Datos.png "Script en JS para formateo y curación de datos")

Vale destacar que algunos ejemplos fueron retirados del Dataset final. Esto se debe a que contenían caracteres inválidos para los predictores que se están manejando o la cantidad de datos superaba lo necesario por cada ejemplo. Esta técnica supone una decisión sobre los datos, la cual debe ser tomada con precaución. En este caso, los ejemplos con problemas de encoding en el archivo fueron omitidos porque suponían un porcentaje muy bajo de la muestra.

| Dataset       	| Ejemplos totales 	| Ejemplos con errores 	| Porcentaje de error 	|
|-------------------|-------------------|-----------------------|-----------------------|
| Cleveland     	| 290              	| 8                    	| 2.75%               	|
| Switzerland   	| 123              	| 0                    	| 0%                  	|
| Hungarian     	| 294              	| 0                    	| 0%                  	|
| Long Beach VA 	| 200              	| 0                    	| 0%                  	|

Ahora tenemos la información correctamente almacenada en 4 archivos CSV distintos. La siguiente imagen muestra el resultado de la ejecución del script para el dataset _"Cleveland"_. También es de utilidad incluir los encabezados de los datos. Los mismos están disponibles en el archivo __.Names__.

![]({{ site.baseurl }}/images/Clevland-Data-Parsed.png "Resultado del archivo Cleveland")

### Unificación de fuentes de datos 

Para cargar el Dataset en Rapidminer, se agregarán las 4 fuentes de datos como archivos al repositorio local. Luego se recuperan en el flujo mediante el operador _Retrieve_. La idea es combinar estos datasets en un único dataset y analizar estadísticas sobre él. Esto se puede realizar aplicando el operador _Join_.

![]({{ site.baseurl }}/images/Combinacion-Dataset.png "Carga inicial de datos en Rapidminer")

Lo primero que se puede apreciar es la gran cantidad de predictores, se dispone de un total de 76. Dentro de la etapa de prepración de los datos también es relevante reducir la cantidad de predictores a utilizar en el modelo. Esto supone un beneficio desde el punto de vista computacional, ya que se deberá trabajar con datos dimensionalmente más simples y por lo tanto con menos requerimientos de memoria y cómputo. Por otra parte, algunos de estos predictores podrían estar correlacionados entre sí, lo cual influye sustancialmente en la estabilidad de algunos modelos. Además de un beneficio, la cantidad de atributos puede ser una restricción para ciertos modelos como K-NN ya que por cada predictor que se agrega, el tiempo de ejecución total aumenta exponencialmente. 

### Análisis de valores faltantes

Por otra parte, múltiples predictores contienen una gran cantidad de valores faltantes. Existen diversas técnicas para combatir los valores faltantes de un dataset, conformando lo que se conoce como imputación de valores. En este caso, usaremos el criterio de valores de faltantes en relación al total de ejemplos para reducir los predictores a utilizar. Esto se debe a que será muy difícil obtener imputaciones semejantes a la realidad cuando los valores faltantes predominan para ese predictor.

La decisión es eliminar aquellos predictores que tengan 33% o más de valores faltantes. Para este caso, si el predictor tiene 300 o más ejemplos con valores faltantes, será directamente eliminado del Dataset. Analizando este criterio, los predictores a eliminar son estos:

| Predictor 	| # Valores Faltantes 	|
|---------------|-----------------------|
| slope     	| 308                 	|
| cigs       	| 420                 	|
| famhist      	| 422                 	|
| rldv5      	| 425                 	|
| years     	| 432                 	|
| thaltime     	| 453                 	|
| thal      	| 477                 	|
| diag      	| 558                 	|
| ramus      	| 567                 	|
| om2       	| 572                 	|
| cathef    	| 588                 	|
| ca        	| 608                 	|
| smoke     	| 669                 	|
| thalsev   	| 769                 	|
| junk       	| 780                 	|
| dm        	| 804                 	|
| thalpul   	| 898                 	|
| restwm    	| 869                 	|
| restef    	| 871                 	|
| exerwm      	| 894                 	|
| exeref    	| 897                 	|
| earlobe     	| 898                 	|
| exerckm   	| 898                 	|
| restckm   	| 899                 	|
| pncaden   	| 899                 	|

Esta operación se puede realizar mediante el operador _Select Attributes_ indicando manualmente los predictores a escoger. Esta decisión nos genera una nueva versión del Dataset con 45 Predictores para trabajar. Realmente serán 44, porque debemos quitar el atributo ID antes de empezar a trabajar. Rapidminer ofrece dos opciones para este tipo de dato, se puede ajustar el rol del predictor a "ID" o se puede directamente eliminar de la tabla. En este caso optaré por eliminarlo ya que no aporta información importante de identificación posterior.

El flujo hasta este punto se ve de esta manera:

![]({{ site.baseurl }}/images/Combinacion-Datasets-start.png "Carga inicial de datos en Rapidminer")

### Imputación de valores

Con respecto a los restantes valores faltantes, aplicaremos una imputación clásica utilizando utilizando K-NN. Este algoritmo basado en distancias, reemplazará los valores que falten recopilando información de ejemplos similares. Vale destacar que el funcionamiento de K-NN parte de la base de que los tipos de datos van a estar bien asignados. En este paso revisamos uno a uno los atributos para asegurarnos de que están bien cargados.

### Detección de outliers
También es importante analizar la distribución de valores para detectar posibles outliers. El primer paso en este análisis es constatar errores de medición o registro de información. Esto se da en el atributo "_Prop_", donde debería existir una variable Binomial, se encuentra un ejemplo que tiene un valor de _22_ para este predictor. Esto claremente se trata de outlier que no aporta información relevante al problema y solamente genera ruido. También sucede para la variable numérica "_Lmt_". Existe un valor de 162 que se encuentra muy alejado del resto de la distribución.

En los siguientes gráficos, en escala logarítmica, se puede apreciar que solamente existe un registro con este valor outlier frente al resto de la distribución.


![]({{ site.baseurl }}/images/Caso-de-Estudio-Analisis-outlier-1.png "Outlier identificado en predictor Prop")

![]({{ site.baseurl }}/images/Caso-de-Estudio-Analisis-outlier-dos.png "Outlier identificado en predictor lmt")

> Para este último caso fue necesario variar la cantidad de bins generadas por el histograma para visualizar el outlier.
>

Estos ejemplos con outliers serán eliminados del Dataset

Los siguientes predictores tuvieron que ser ajustados en términos del tipo de dato:

| Predictor 	| Tipo de dato nuevo 	|
|---------------|-----------------------|
| dig       	| Binomial            	|
| diuretic    	| Binomial             	|
| exang     	| Binomial           	|
| painloc     	| Binomial           	|
| painexer     	| Binomial           	|
| relrest     	| Binomial           	|
| prop      	| Binomial           	|
| nitr       	| Binomial           	|
| htn       	| Binomial           	|
| xhypo     	| Binomial           	|
| pro       	| Binomial           	|
| sex       	| Binomial           	|
| num       	| Nominal           	|
| restcg       	| Nominal           	|
| cp        	| Nominal           	|

### Feature selection

Para este caso, se ha decidido no utilizar técnicas de Feature Selection avanzadas como Análisis de Componentes Lineales. Por el contrario, se seleccionarán los predictores cuya correlación con la variable objetivo sea alta. Para ello, se definirá un coeficiente de umbral para evaluar si un predictor será utilizado, a partir de su valor de correlación. Es importante entender que este filtro servirá exclusivamente para predictores numéricos. El resto de los atributos permanecerán en el Dataset.

La correlación con la variable de salida puede verse por medio de la matriz de correlación. Tomando una correlación de por lo menos 0.14, es decir mayor a 0.14 y menos a -0.14, el conjunto de predictores que cumplen esta condición es este:

- thalach
- age
- thaldur
- proto
- chol
- thalrest
- oldpeak
- laddist
- ladprox
- lvx4
- met
- lmt
- lfv
- lvx3
- painexer
- rcadist
- tpeakbps
- tpeakbpd


Ahora que tenemos el conjunto final de predictores a utilizar, es pertinente evaluar las distintas distribuciones que puedan tener. Este dato es importante especialmente para ciertos algoritmos que tienen un mejor rendimiento con distribuciones gaussianas. También es una oportunidad para evaluar la proporción de clases de la variable objetivo.

### Transformaciones sobre los datos

![]({{ site.baseurl }}/images/Caso-de-estudio-Proporcion-de-clase.png "Representación de clases de la variable objetivo")

Claramente la primera clase tiene una sobrerepresentación en el Dataset. Los ejemplos que corresponden a esta clase son los pacientes cuya predicción indica que no sufrieron enfermedades cardíacas.

En los predictores seleccionados de tipo continuo no se observa un sesgo o __Skewness__ muy pronunciado en ningún sentido. Por lo tanto, considero que los datos sin trasformaciones podrían ser utilizados en primera instancia para evaluar distintos modelos. De ser necesario ajustar alguno de ellos, existen diversas técnicas para transformar estos predictores. La raíz cuadrada, el logaritmo natural y la función inversa son probablemente las más utilizadas para obtener distribuciones más apropiadas, es decir, más parecidas a la normal.

## Modelo

### Elección del modelo

La generación del modelo depende en gran parte del tipo de problema que estemos atacando, esto implica lo que queremos conseguir con el modelo y también la naturaleza del problema. Para comenzar, estamos frente a un problema de clasificación supervisada. Esto se debe a que cada ejemplo en nuestro Dataset está etiquetado con el valor de una variable objetivo dada. 

El objetivo del modelo será abstraer la relación existente entre los datos de entrada y la salida a partir de los valores presentes en el Dataset. Además, estamos buscando ubicar los nuevos registros en cada una de las clases de la variable objetivo, que en este caso son 5 distintas, eso significa que estamos ante un problema de clasificación multivariable.

Ante esta situación, creo que una buena decisión sería comenzar con un Árbol de Decisión. Luego de evaluar el modelo, optaría por avanzar a un modelo de ensamble de Random Forest y revisar las métricas con respecto al modelo anterior.

### Separación de datos para Entrenamiento y Test
El entrenamiento del modelo se realizará con una parte mayoritaria del Dataset. Esto implica una división donde una parte se utilizará en el entrenamiento y el resto de ejemplos para probar el modelo generado. En este caso, optaré por seguir la regla del __70 / 30__. La partición del Dataset puede ser realizada en Rapidminer mediante el operador "_Split Data_".

La selección de ejemplos para cada una de las particiones debe tener un criterio, esta configuración se conoce como _Sampling Type_. En principio se ofrecen varias opciones en el operador de _Split_, las más recomendables son muestreo aleatorio y muestro estratificado. En este caso usaremos el estratificado para mantener la proporción de clases del Dataset original en estas nuevas particiones.

El flujo debería verse de la siguiente manera:

![]({{ site.baseurl }}/images/Flujo-pre-modelo.png "Flujo de Rapidminer antes del entrenamiento")

El entrenamiento se realizará aplicando un K-Fold Cross Validation utilizando un valor de K estándar de 10. Esta técnica permite general un modelo más realista, aprovechando la información que aporta el Dataset al máximo. Esto se logra ejecutando distintas versiones del modelo y optimizando una métrica en particular. El proceso divide el dataset en K partes iguales y entrena usando K - 1 partes en cada iteración. Asegurar la independencia de los datasets de test es un factor clave para el éxito de esta técnica.

Dentro del operador de Cross Validation, ubicamos el operador de _Árbol de decisión_ con los parámetros por defecto. En principio esto servirá para evaluar la performance general del modelo y encaminar una mejora del modelo utilizando este algoritmo de clasificación.

El modelo debería verse de la siguiente manera:

![]({{ site.baseurl }}/images/Caso-de-Estudio-Modelo-Arbol-Decision.png "Flujo de Rapidminer final")

### Evaluación del Árbol de Decisión

Los resultados preliminares de performance indican un rendimiento aceptable. El principal componente a analizar en esta fase preliminar es la Matriz de Confusión. Aquí se presentan las distintas clases a predecir y la correspondencia con los valores predichos. De esta tabla pueden obtenerse los falsos positivos, los falsos negativos, los verdaderos positivos y verdaderos negativos. En el caso ideal, todos los valores predichos corresponderán con los valores reales, el modelo tendrá un 100% de precisión y también la matriz de confusión tendrá valores distintos de cero exclusivamente en la diagonal principal.

![]({{ site.baseurl }}/images/Caso-de-Estudio-Matriz-Confusion-2.png "Matriz de confusión para modelo de Árbol de decisión")

Por medio de la matriz de confusión, se pueden obtener también métricas interesantes a analizar en un problema multiclase. La más destacada se llama __Class Recall__ la proporción de casos en que el modelo predijo bien esa clase en particular. Este dato resulta importante ya que nos indica el nivel de funcionamiento del modelo para una clase en particular.

En este caso de estudio, la variable objetivo tendrá un valor entero entre 0 y 4, siendo las clases extremas las más relevantes. Aquellos pacientes que tengan un 0 en la predicción, tendrán una probabilidad muy baja de tener una enfermedad cardíaca. Lo contrario sucede para aquellos que obtienen 3 o 4, donde es muy importante que la situación sea detectada para poder tratar y prevenir mayores consecuencias. 

El peor _Recall_ obtenido es para la clase 2 que esté en medio, con un 16.28%. Sin embargo, esto no parecería ser tan crítico porque la mayoría de las predicciones erradas dieron un resultado entre 1 y 3, valores no muy extremos que se encuentran en esa zona intermedia de riesgo.

Debemos recordar que al principio nos encontramos con un Dataset desbalanceado, existían muchos más ejemplos de la clase 0 que de las demás clases. Esto es peligroso al momento de validar un modelo porque sesga el resultado de __Accuracy__. Si el modelo clasifica todos los ejemplos como la clase predominante, el error de algunos ejemplos (las clases menos representadas) significarán poco con respecto al total. Una herramienta útil para estas situaciones es el balanceo por pesos, donde los registros del entrenamiento se ponderan según la clase que tienen asignada. Este criterio de ponderación busca igualar la representación de clases en los ejemplos del Dataset.

### Comparación con Random Forest

La mayor parte de los modelos de producción utilizan un enfoque conocido como _Ensambles_. Esta técnica tiene como objetivo reducir los errores de un modelo en particular con respecto a un dataset en particular. Es posible que la solución encontrada aprenda demasiado de un Dataset y temrine asimilando el ruido de los datos como información útil. Si bien localmente puede que la solución sea útil, no lo será en la generalidad del espacio de posibilidades. Entonces, resulta úlil evaluar la _opinión_ que distintos modelos puedan tener para una predicción en particular. 

De esta manera, se generan distintos algoritmos para elegir un resultado como respuesta, en base a las respuestas emitidas por los modelos individuales. Esto se puede generar de distintas maneras. Se podría tener una colección de modelos variando las siguientes características:

- Ejemplos para entrenamiento del modelo
- Algoritmo del modelo
- Parámetros del modelo
- Predictores del Dataset

En este caso, aplicaré un __Random Forest__ para evaluar si el rendimiento general del modelo obtiene una mejora. Para ello, reemplazo el operador de _Árbol de decisión_ por el de Random Forest y configuro para que utilice un tamaño de 1000 árboles y el criterio de votación sea por confianza.

Estos son los resultados de la ejecución de ese modelo:

![]({{ site.baseurl }}/images/Caso-de-Estudio-Matriz.Confusión-3.png "Matriz de confusión para modelo de Random Forest")

Como se puede ver en la matriz de confusión, los distintos Class Recall mejoraron bastante en comparación con los resultados del árbol de decisión. Esto me aporta mayor confianza a la hora de utilizar este modelo de ensamble. 

También vemos incrementada la precisión del modelo en general, ahora tenemos un 81.04%.

