import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



class naive_bayes_st:
    def __init__(self, database, test_size=0.2):
        self.database = database
        self.test_size = test_size
        self.desc = r'''
        # **Naive Bayes**

        Particularmente, este algoritmo no lo conocia, y por lo que he visto hasta ahora funciona como un **clasificador** basandose principalmente en el **teorema de bayes**.

        **Teorema de bayes**

        $$
        P(A/B) = \frac{P(B/A) \cdot P(A)}{P(B)}
        $$

        Eso si, para aprovechar este teorema es que se tiene que cumplir la condicion de que los atributos o **componentes del vector X sean independientes entre si (Se asume que los eventos son independientes)**.

        $$
        P(y/X) = \frac{P(X/y) \cdot P(y)}{P(X)} = \frac{P(x_{1}/y) \quad ... \quad P(x_{n}/y) \cdot P(y)}{P(X)}
        $$

        Asi, luego la manera de escoger a que clasificacion pertenece el vector X, es calculando todas las probabilidades condicionales (**Nota**: el $P(x)$ lo podemos omitir ya que va a estar presente en todas las ecuaciones)


        $$
        y = argmax_{y} \quad P(x_{1}/y) \quad ... \quad P(x_{n}/y) \cdot P(y)
        $$

        $$
        y = argmax_{y} \quad log(P(x_{1}/y)) + \quad ... \quad + log(P(x_{n}/y)) + log(P(y))
        $$


        **Por ultimo, nos falta definir:**

        $P(y)$: Frecuencia (cantidad de veces que esta presente la clasificacion y en los datos)

        $$
        P(x_{i}/y) = \frac{1}{\sqrt{2\pi \sigma_{y}^{2}}} \cdot e^{(-\frac{(x_{i} - \mu_{y})^2}{2σ_{y}^{2}})}
        $$
        '''

    def solve(self):
        self.X, self.y = self.database.data, self.database.target
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=1234)
        self.sklearn_clf = GaussianNB()
        self.sklearn_clf.fit(X_train, y_train)
        y_pred = self.sklearn_clf.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        st.metric('Acierto', value=f'{np.round(acc, 2)*100}%')

    def visualization(self):
        n_features = int(self.database.data.shape[1])
        self.x_feature = st.slider('Variables en eje x', 1, n_features, 1)
        self.y_feature = st.slider('Variables en eje y', 1, n_features, 2)

        self.X = np.c_[self.database.data[:, self.x_feature-1:self.x_feature], self.database.data[:, self.y_feature-1:self.y_feature]]
        self.y = self.database.target
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=1234)
        self.sklearn_clf = GaussianNB()
        self.sklearn_clf.fit(X_train, y_train)

        x1_min, x1_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        x2_min, x2_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        h = 0.02 # Salto que vamos dando
        x1_i = np.arange(x1_min, x1_max, h)
        x2_i = np.arange(x2_min, x2_max, h)
        x1_x1, x2_x2 = np.meshgrid(x1_i, x2_i)
        y_pred = self.sklearn_clf.predict(np.c_[x1_x1.ravel(), x2_x2.ravel()])
        y_pred = y_pred.reshape(x1_x1.shape)

        plt.figure(1, figsize=(4, 3))
        plt.pcolormesh(x1_x1, x2_x2, y_pred, cmap=plt.cm.Paired)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlim(x1_x1.min(), x1_x1.max())
        plt.ylim(x2_x2.min(), x2_x2.max())
        return plt.gcf()
