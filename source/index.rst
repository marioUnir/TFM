=======================================
Documentación del TFM - UNIR
=======================================

.. image:: _static/unir_logo.png
   :alt: UNIR Logo
   :width: 300px
   :align: center

Bienvenido a la documentación de la librería **credit-card**, desarrollada como parte
del Trabajo de Fin de Máster (TFM) en la Universidad Internacional de La Rioja (UNIR).

Esta librería ha sido creada por Cristina Domínguez y Mario Río, con el objetivo de proporcionar 
una  herramienta integral para la **predicción del incumplimiento de pagos (default)** en 
tarjetas de crédito. La implementación se basa en **técnicas avanzadas de machine learning** 
y análisis de datos, estructurada en módulos que cubren desde la exploración inicial 
de datos hasta la calibración y evaluación de modelos de clasificación.

Introducción
------------

La librería **credit-card** está diseñada para facilitar el análisis y modelado de 
riesgo crediticio, en particular la predicción del incumplimiento de pagos (default) 
de clientes de tarjetas de crédito. A través de diferentes módulos, esta librería 
proporciona herramientas para el análisis exploratorio de datos (EDA), detección de 
valores atípicos, preprocesamiento de datos y modelado predictivo.

El objetivo principal es ofrecer una solución estructurada que permita a los 
profesionales del análisis de datos y del riesgo financiero optimizar sus procesos 
de calibración y evaluación de modelos de clasificación para predicción de default.

Módulos principales
-------------------

La librería está dividida en los siguientes módulos:

- **EDA**: Exploración de datos, identificación de patrones, estadísticas descriptivas 
  y análisis de correlaciones.
- **Outliers**: Detección de valores atípicos mediante métodos estadísticos y de 
  machine learning (IQR, Isolation Forest, LOF, Gaussian Mixture Model).
- **Preprocessing**: Preprocesamiento de datos con imputación de valores faltantes, 
  codificación de variables categóricas y escalado de variables numéricas.
- **Modelling**: Construcción y evaluación de modelos de clasificación utilizando 
  algoritmos como regresión logística, árboles de decisión, random forest y gradient 
  boosting, con búsqueda de hiperparámetros mediante validación cruzada.

Contenido
---------

.. toctree::
   :maxdepth: 2

   EDA
   Outliers
   Preprocessing
   Modelling
