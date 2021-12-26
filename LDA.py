import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

class LDA_st:
    def __init__(self, database, test_size=0.2):
        self.database = database
        self.test_size = test_size
        self.desc = r'''
        # **LDA (Linear Discrimination Analysis)**

        **Objetivo**

        Reducir el numero de variables (**features**).
        El objetivo es proyectar un conjunto de datos a un espacio dimensional mas reducido. (Similar a como se hacia con **PCA**)

        **PCA vs LDA**

        - **PCA**: Encontrar los ejes que maximizan la varianza en los datos.
        - **LDA**: El interes esta puesto en los ejes que maximizan la separacion entre clases de datos.
        - **LDA**: es un tipo de **aprendizaje supervizado** (utiliza la clasificacion (etiquetas) de los datos para entrenar al algortimo), en cambio **PCA** es un tipo de **aprendizaje no supervizado** (sin etiquetas)

        **Within-class scatter matrix**

        $$
        S_{w} = \sum_{c} S_{c}
        $$

        $$
        S_{c} = \sum_{i \in c} (x_{i} - \bar{x_{c}}) \cdot (x_{i} - \bar{x_{c}})^{T}
        $$

        **Between class scatter matrix**

        $$
        S_{B} = \sum_{c} η \cdot (\bar{x_{c}} - \bar{x}) \cdot (\bar{x_{c}} - \bar{x})^{T}
        $$

        **Vectores y valores propios**

        Calcular los vectore y valores propios de la siguiente matriz:

        $$
        S_{W}^{-1} S_{B}
        $$


        **Metodo**

        - Calcular $S_{B}$
        - Calcular $S_{W}$
        - Calcular los vecores y valores propios de $S_{W}^{-1} S_{B}$
        - Ordenar los vectores propios en funcion de los valores propios de manera decreciente
        - Escoger los primeros k vectores propios los cuales vendran a representar las nuevas k dimensiones
        - Transformar los datos en las nuevas dimensiones (**se hace con producto punto**) '''

        self.x_feature = 1
        self.y_feature = 2
        self.n_components = 2

    def params(self):
        n_features = int(self.database.data.shape[1])
        self.n_components = st.slider('Numero de componentes', 1, n_features, 2)

    def solve(self):
        self.x_feature = st.slider('Componente eje x', 1, self.n_components, 1)
        self.y_feature = st.slider('Componente eje y', 1, self.n_components, 2)
        X = self.database.data
        y = self.database.target
        sklearn_clus = LinearDiscriminantAnalysis(n_components=self.n_components)
        sklearn_clus.fit(X, y)
        X_proyected_sk = sklearn_clus.transform(X)

        x1 = X_proyected_sk[:, self.x_feature-1]
        x2 = X_proyected_sk[:, self.y_feature-1]

        plt.figure(figsize=(12, 8))
        plt.scatter(x1, x2, c=y, edgecolors='none', alpha=0.8, cmap=plt.cm.get_cmap('viridis', len(y)))
        plt.xlabel(f'Componente {self.x_feature}')
        plt.ylabel(f'Componente {self.y_feature}')
        plt.colorbar()

        return plt.gcf()
