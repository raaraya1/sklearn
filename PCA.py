import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.decomposition import PCA as PCA_sk
import matplotlib.pyplot as plt

class PCA_st:
    def __init__(self, database, test_size=0.2):
        self.database = database
        self.test_size = test_size
        self.desc = r'''
        # **PCA (Principal Component Analysis)**

        El objetivo principal con este metodo es definir una nueva dimension para el set de datos (siendo estas nuevas dimensiones ortogonales y por tanto independientes).

        **Varianza**

        $$
        var(X) = \frac{1}{n} \sum (X_{i} - \bar{X})^2
        $$

        **Matriz de Covarianzas**

        $$
        Cov(X, Y) = \frac{1}{n} \sum (X_{i} - \bar{X})(Y_{i} - \bar{Y})^T
        $$

        $$
        Cov(X, X) = \frac{1}{n} \sum (X_{i} - \bar{X})(X_{i} - \bar{X})^T
        $$

        **Valores y Vectores Propios**

        Los vectores propios apuntan en la direccion donde se genera la maxima varianza y el correspondiente valor propio indica el grado de importancia del vector.

        $$
        A \vec{v} = λ \vec{v}
        $$

        **Metodo**
        - Sustraer al vector X su media.
        - Calcular la Cov(X, X)
        - Calcular los vectores y valores propios de las matrices de covarianza
        - Ordenar los vectores propios segun su importancia (en base a su valor propio) en orden decreciente
        - Escoger los primeros k vectores propios y estos pasaran a ser las nuevas k dimensiones
        - Por ultimo, transformar (proyectar) los datos en las nuevas dimensiones (esto se hace con un producto punto)'''

        self.x_feature = 1
        self.y_feature = 2
        self.n_components = 2

    def params(self):
        n_features = int(self.database.data.shape[1])
        self.n_components = st.slider('Numero de componentes', 1, n_features, 2)
        self.x_feature = st.slider('Componente eje x', 1, self.n_components, 1)
        self.y_feature = st.slider('Componente eje y', 1, self.n_components, 2)

    def solve(self):
        X = self.database.data
        y = self.database.target
        sklearn_clus = PCA_sk(n_components=self.n_components)
        sklearn_clus.fit(X)
        X_proyected_sk = sklearn_clus.transform(X)

        x1 = X_proyected_sk[:, self.x_feature-1]
        x2 = X_proyected_sk[:, self.y_feature-1]

        plt.scatter(x1, x2, c=y, edgecolors='none', alpha=0.8, cmap=plt.cm.get_cmap('viridis', 3))
        plt.xlabel(f'Componente {self.x_feature}')
        plt.ylabel(f'Componente {self.y_feature}')
        plt.colorbar()

        #fig = plt.show().get_fig()
        return plt.gcf()
