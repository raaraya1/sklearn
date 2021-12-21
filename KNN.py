import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class KNN_st:
    def __init__(self, database, test_size=0.2):
        self.database = database
        self.test_size = test_size
        self.desc = '''
                        # **Algoritmo KNN (K Nearest Neighbor)**

                Este algoritmo se basa en que para predecir una clasificación sobre un nuevo dato, lo primero que debemos hacer es calcula la **distancia euclidiana** con el resto de los datos, **seleccionar los k datos con menor distancia** (mas cercanos) y por ultimo **asignar la clasificación en funcion a la moda** (categoria mas repetida) de esos k datos seleccionados.

                **Distancia Euclidiana**

                $$
                Dist= \sqrt(\sum_{i=1}^n (Xtest_{i} - Xtrain_{i})^2))
                $$

                '''
        self.neighbors = 5

    def params(self):
        self.neighbors = st.slider('Numero de vecinos',
                                    min_value=0,
                                    max_value=15,
                                    value=5)

    def solve(self):
        X, y = self.database.data, self.database.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
        sklearn_clf = KNeighborsClassifier(self.neighbors)
        sklearn_clf.fit(X_train, y_train)
        y_pred = sklearn_clf.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        st.metric('Acierto', value=f'{np.round(acc, 2)*100}%')
