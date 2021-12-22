import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

class ICA_st:
    def __init__(self, database, test_size=0.2):
        self.database = database
        self.test_size = test_size
        self.desc = r'''
        # **ICA (Independent Component Analysis)**

        Fuente: https://towardsdatascience.com/independent-component-analysis-ica-in-python-a0ef0db0955e


        ICA es un metodo que se utiliza para identificar las componentes de una señal multivariada. De esta manera es que podemos extraer un componente que se encuentre mezclados con otros.

         - A $X$ restarle su media $\bar{X}$
         - Transformar $X$ de manera que las potenciales correlaciones entre las componentes sea removidas y que la varianza para cada componente sea igual a 1. (Hacer que la matriz de covarianza se paresca a la matriz de identidad)

          $$
          \hat{x} = E \cdot \sqrt{D} \cdot E^{T} \cdot x
          $$

           - $D$: Diagonal con valores propios (de la matriz de covarianzas)
           - $E$: Matrix con vectores propios (de la matriz de covarianzas)

         - Escoger valores aleatorios para armar la matriz $W$.
         - Calcular los nuevos valores para $W$

          $$
          w_{i} = \frac{1}{n} \sum X \cdot tanh(W^{T} \cdot X) - \frac{1}{n} \sum X \cdot (1 - tanh^{2}(W^{T} \cdot X) \cdot W)
          $$

        $$
        w_{i} = w_{i} - \sum_{j=1}^{p-1} (w_{p}^{T}w_{j})w_{j}
        $$

         - Normalizar $w_{p}$

        $$
        w_{p} = \frac{w_{p}}{||w_{p}||}
        $$

         - Chequear condicion de termino. Si no se cumple volvemos a calcular los nuevos valores de $w$


        $$
        w_{p}^{T}w_{p+1} - 1 < Tolerance
        $$

         - Calcular la fuentes independientes como $S = W \cdot X$'''

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
        sklearn_clus = FastICA(n_components=self.n_components)
        X_proyected_sk = sklearn_clus.fit_transform(X)

        x1 = X_proyected_sk[:, self.x_feature-1]
        x2 = X_proyected_sk[:, self.y_feature-1]

        plt.scatter(x1, x2, c=y, edgecolors='none', alpha=0.8, cmap=plt.cm.get_cmap('viridis', len(y)))
        plt.xlabel(f'Componente {self.x_feature}')
        plt.ylabel(f'Componente {self.y_feature}')
        plt.colorbar()

        return plt.gcf()
