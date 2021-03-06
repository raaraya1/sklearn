import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt


class agglomerative_clustering_st:
    def __init__(self, database, test_size=0.2):
        self.database = database
        self.test_size = test_size
        self.desc = r'''
        # **Agglomerative Clustering**

        Agglomerative Clustering es un tipo de algoritmo que agrupa de manera jerárquica. De esta manera lo que se hace es considerar a cada observación como un cluster y luego ir juntando aquellos que sean más similares. Esto lo repetimos hasta alcanzar un numero de clusters deseado.

        **Método**
         - Inicializamos todos los puntos como clusters
         - Tomamos dos clusters que se encuentren cercanos y los unificáramos en un único cluster.
         - Repetimos el paso anterior hasta conseguir un numero de clusters deseado.

        **Criterios para medir la similitud entre clusters**

         - Distancia entre los puntos **más cercanos** de dos clusters distintos.
         - Distancia entre los puntos **más lejanos** de dos clusters distintos.
         - Distancia entre los promedios de cada cluster.

        '''
        self.x_feature = 1
        self.y_feature = 2
        self.n_clusters = 3

    def params(self):
        n_targets = len(set(self.database.target))
        self.n_clusters = st.slider('Numero de clusters', 1, n_targets, 1)

    def solve(self):
        n_features = int(self.database.data.shape[1])
        self.x_feature = st.slider('Variables en eje x', 1, n_features, 1)
        self.y_feature = st.slider('Variables en eje y', 1, n_features, 2)
        X = self.database.data
        sklearn_clus = AgglomerativeClustering(self.n_clusters, linkage='single')
        pred = sklearn_clus.fit_predict(X)
        fig, ax = plt.subplots(figsize=(12,8))
        ax.scatter(X[:, self.x_feature-1], X[:, self.y_feature-1], c=pred)
        plt.title(f'{self.n_clusters} Clusters')
        return fig
