import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



class KNN_st:
    def __init__(self, database, test_size=0.2):
        self.database = database
        self.test_size = test_size
        self.desc = '''
                        # **Algoritmo KNN (K Nearest Neighbor)**

                Este algoritmo se basa en que, para predecir una clasificación sobre un nuevo dato, lo primero que debemos hacer es calcula la **distancia euclidiana** con el resto de los datos, **seleccionar los k datos con menor distancia** (más cercanos) y por ultimo **asignar la clasificación en función a la moda** (categoría más repetida) de esos k datos seleccionados.

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
        self.X, self.y = self.database.data, self.database.target
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=1234)
        self.sklearn_clf = KNeighborsClassifier(self.neighbors)
        self.sklearn_clf.fit(X_train, y_train)
        y_pred = self.sklearn_clf.predict(X_test)
        acc = accuracy_score(y_pred, y_test)

        c1, c2 = st.columns([4, 1])
        c2.metric('Acierto', value=f'{np.round(acc, 2)*100}%')
        df = pd.DataFrame(confusion_matrix(y_pred, y_test))
        labels = self.database.target_names
        df.columns = labels
        df.index = labels
        c1.write('**Confusion Matrix**')
        c1.dataframe(df)

    def visualization(self):
        n_features = int(self.database.data.shape[1])
        self.x_feature = st.slider('Variables en eje x', 1, n_features, 1)
        self.y_feature = st.slider('Variables en eje y', 1, n_features, 2)

        self.X = np.c_[self.database.data[:, self.x_feature-1:self.x_feature], self.database.data[:, self.y_feature-1:self.y_feature]]
        self.y = self.database.target
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=1234)
        self.sklearn_clf = KNeighborsClassifier(self.neighbors)
        self.sklearn_clf.fit(X_train, y_train)

        x1_min, x1_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        x2_min, x2_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        h = 0.02 # Salto que vamos dando
        x1_i = np.arange(x1_min, x1_max, h)
        x2_i = np.arange(x2_min, x2_max, h)
        x1_x1, x2_x2 = np.meshgrid(x1_i, x2_i)
        y_pred = self.sklearn_clf.predict(np.c_[x1_x1.ravel(), x2_x2.ravel()])
        y_pred = y_pred.reshape(x1_x1.shape)

        plt.figure(1, figsize=(12, 8))
        plt.pcolormesh(x1_x1, x2_x2, y_pred, cmap=plt.cm.Paired)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlim(x1_x1.min(), x1_x1.max())
        plt.ylim(x2_x2.min(), x2_x2.max())
        return plt.gcf()
