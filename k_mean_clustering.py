import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.cluster import KMeans as KM
import matplotlib.pyplot as plt


def plot(X, clusters, centroids, x_feature, y_feature):
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, index in enumerate(clusters):
      x = X[index].T[x_feature-1]
      y = X[index].T[y_feature-1]
      point = np.array([x, y])
      ax.scatter(*point)

    for point in centroids:
        x = point[x_feature-1]
        y = point[y_feature-1]
        points = np.array([x, y])
        ax.scatter(*points, marker="o", linewidth=15)

    return fig

class k_mean_clustering_st:
    def __init__(self, database, test_size=0.2):
        self.database = database
        self.test_size = test_size
        self.desc = r'''
        # **K-Mean Clustering**

        El objetivo en esta ocasión es segmentar información desclasificada (**unsupervised learning**)

        Así, este método asigna a una muestra de datos una clase en base a la distancia promedio entre los datos.

        **Iterative Optimization**

        - Inicializamos los centros de manera aleatoria
        - Iteramos hasta converger
         - Actualizamos las clasificaciones de los datos utilizando el centroide.
         - Actualizamos el centroide. (este corresponde a la posición del centro para una clase)


        **Distancia entre vectores**

        $$
        d(p, q) = \sqrt{\sum (p_{i} - q_{i})^{2}}
        $$
        '''
        self.x_feature = 1
        self.y_feature = 2
        self.n_clusters = 3
        self.max_iter = 150

    def params(self):
        self.n_features = int(self.database.data.shape[1])
        self.n_clusters = st.slider('Numero de segmentos', 1, 10, 3)
        self.max_iter = st.slider('Numero maximo de iteraciones', 100, 200, 150)

    def solve(self):
        self.x_feature = st.slider('Variables en eje x', 1, self.n_features, 1)
        self.y_feature = st.slider('Variables en eje y', 1, self.n_features, 2)
        X = self.database.data
        sklearn_clus = KM(n_clusters=self.n_clusters, max_iter=self.max_iter)
        sklearn_clus.fit(X)
        pred = sklearn_clus.predict(X)
        classes = np.unique(pred)
        clusters = [[] for i in classes]
        for idx, value in enumerate(pred):
          clusters[value].append(idx)

        return plot(X=X,
                    clusters=clusters,
                    centroids=sklearn_clus.cluster_centers_,
                    x_feature=self.x_feature,
                    y_feature=self.y_feature)
