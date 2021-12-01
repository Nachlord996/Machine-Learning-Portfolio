---
toc: true
layout: post
description: ""
categories: []
title: "Caso de Estudio: Detección de enfermedad cardíaca"
sticky_rank: 1
---
## Contexto

Me resulta particularmente interesante las posibilidades de aplicación de la Ciencia de Datos en el área de la Medicina. Los Hospitales y centros médicos en general, son instituciones que generan y registran información sobre el estado de sus pacientes diariamente. Por lo que, en principio es un buen campo de investigación para proyectos de esta índole.

La información puede ser aprovechada mediante técnicas de Machine Learning para el entrenamiento de modelos que aporten valor a la labor médica. Sin embargo, la mayoría de las instituciones no estarán dispuestas a compartir dicha información para proteger así la privacidad de sus usuarios. En definitiva, los pacientes depositan su confianza en estos centros médicos y tienen el derecho legítimo de que su información personal no sea filtrada. Este derecho se ve respaldado por la obligación, del otro lado del mostrador, de resguardar esta información. Esto último deriva en que la disponibilidad de información sea mayormente escasa y difícil de conseguir.

En este caso de estudio se pretende analizar información de pacientes para la detección de enfermedades cardíacas. Se parte de Datasets que están disponibles públicamente y han sido objeto de estudio en varios artículos sobre esta temática. Estos datos provienen de 4 bases de datos distintas de:

 1. Cleveland Clinic Foundation
 1. Hungarian Institute of Cardiology, Budapest
 1. V.A. Medical Center, Long Beach, CA
 1. University Hospital, Zurich, Switzerland  

El objetivo principal es demostrar las habilidades aprendidas considerando todas las etapas de un proyecto de Machine Learning.

## Preparación de los Datos

A pesar de que los Datasets tienen orígenes distintos, los atributos que los componen se mantienen en todos los Datasets. Por lo tanto, podemos afirmar que una posible unificación de los datos será correcta y beneficiosa para el estudio, ya que contaremos con mayor cantidad de ejemplos y todos tienen la misma estructura. Esto implica también, que los datos registrados estén en las _mismas unidades_ y _encodeados_ de la misma manera. Diversos tipos de datos pueden tener el mismo significado pero tener distinta representación. Durante la preparación de los datos, debemos prestar especial atención a este aspecto ya que es uno de los errores más frecuentes y es fácil pasar por alto.

Como primera aproximación, deberemos investigar la estructura de los archivos, para luego poder levantarlos correctamente en la herramienta escogida.

> Cada uno de los extractos de las bases de datos estan en archivos individuales __.DATA__. Existe además un archivo __.NAMES__ que detalla la información presente en los anteriormente mencionados.
> 

![]({{ site.baseurl }}/images/Validacion-preparacion-de-datos.png "Contenido del archivo Cleveland.data")

La estructura de los datos no parece ser la más conveniente para su lectura, por lo que surge la necesidad de realizar transformaciones de formato. Para ello, realizaremos un script que convierta cada Dataset a un archivo __.csv__ independiente.

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

Para cargar el Dataset en Rapidminer, se agregarán las 4 fuentes de datos como archivos al repositorio local. Luego se recuperan en el flujo mediante el operador _Retrieve_. La idea es combinar estos datasets en un único dataset y analizar estadísticas sobre él. Esto se puede realizar aplicando el operador _Join_.

![]({{ site.baseurl }}/images/Combinacion-Datasets.png "Carga inicial de datos en Rapidminer")

Lo primero que se puede apreciar es la gran cantidad de predictores, se dispone de un total de 76. Dentro de la etapa de prepración de los datos también es relevante reducir la cantidad de predictores a utilizar en el modelo. Esto supone un beneficio desde el punto de vista computacional, ya que se deberá trabajar con datos dimensionalmente más simples y por lo tanto con menos requerimientos de memoria y cómputo. Por otra parte, algunos de estos predictores podrían estar correlacionados entre sí, lo cual influye sustancialmente en la estabilidad de algunos modelos. Además de un beneficio, la cantidad de atributos puede ser una restricción para ciertos modelos como K-NN ya que por cada predictor que se agrega, el tiempo de ejecución total aumenta exponencialmente. 

Por otra parte, múltiples predictores contienen una gran cantidad de valores faltantes. Existen diversas técnicas para combatir los valores faltantes de un dataset, conformando lo que se conoce como imputación de valores. En este caso, usaremos el criterio de valores de faltantes en relación al total de ejemplos para reducir los predictores a utilizar. Esto se debe a que será muy difícil obtener imputaciones semejantes a la realidad cuando los valores faltantes predominan para ese predictor.

La decisión es eliminar aquellos predictores que tengan 10% o más de valores faltantes. Para este caso, si el predictor tiene 90 o más ejemplos con valores faltantes, será eliminado del Dataset. Analizando este criterio, los predictores a eliminar son estos:

| Predictor 	| # Valores Faltantes 	|
|---------------|-----------------------|
| fbs       	| 90                  	|
| met       	| 105                 	|
| proto     	| 112                 	|
| eldv5e    	| 142                 	|
| ramus     	| 235                 	|
| laddist   	| 236                 	|
| rcadist   	| 245                 	|
| diag      	| 246                 	|
| lvx1      	| 270                 	|
| om2       	| 271                 	|
| ladprox   	| 275                 	|
| painexer  	| 282                 	|
| painloc   	| 286                 	|
| relrest   	| 308                 	|
| slope     	| 420                 	|
| cigs      	| 422                 	|
| famhist   	| 425                 	|
| rldv5     	| 432                 	|
| years     	| 453                 	|
| thaltime  	| 477                 	|
| thalsev   	| 558                 	|
| cxmain    	| 567                 	|
| om1       	| 572                 	|
| rcaprox   	| 588                 	|
| junkca    	| 608                 	|
| smoke     	| 669                 	|
| thalpul   	| 769                 	|
| dm        	| 804                 	|
| earlobe   	| 855                 	|
| restwm    	| 869                 	|
| restef    	| 871                 	|
| thal      	| 894                 	|
| exerwm    	| 897                 	|
| cmo       	| 898                 	|
| exerckm   	| 898                 	|
| restckm   	| 899                 	|
| pncaden   	| 899                 	|



Para este caso, se ha decidido no utilizar técnicas de Feature Selection avanzadas como Análisis de Componentes Lineales. Por el contrario, primero se seleccionarán los predictores cuya correlación con la variable objetivo sea alta. Para ello, se definirá un coeficiente de umbral para evaluar si un predictor será utilizado, a partir de su valor de correlación. 






