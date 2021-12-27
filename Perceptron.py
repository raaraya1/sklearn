import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron as P_sk
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class perceptron_st:
    def __init__(self, database, test_size=0.2):
        self.database = database
        self.test_size = test_size
        self.desc = r'''
        # **Perceptron**

        Este es el modelo más sencillo y que sirve de introducción a los modelos de redes neuronales. En particular, su funcionamiento es bastante similar al modelo de regresión linear. con la diferencia de que ocupa una función de activación en la salida (**función no lineal**).

        **Modelo Lineal**

        $$
        f(w, b) = w^{t}x + b
        $$

        **Función de Activación**

        $$
        z(x) \in (0, 1) \quad si \quad x \geq 0
        $$

        **Aproximación (predicción)**

        $$
        \hat{y} = z(w^{t}x + b)
        $$

        **Reglas de actualización (aquí se encuentra incluido el bias)**

        $$
        w = w + \Delta w = w + lr(y_{i} - \hat{y_{i}})x_{i}
        $$
        '''

    def solve(self):
        self.X, self.y = self.database.data, self.database.target
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=1234)
        self.sklearn_regr = P_sk(random_state=1234)
        self.sklearn_regr.fit(X_train, y_train)
        y_pred = self.sklearn_regr.predict(X_test)
        acc = mean_squared_error(y_pred, y_test)
        st.metric('MSE (Mean Square Error)', value=f'{np.round(acc, 2)}')

    def visualization(self):
        n_features = int(self.database.data.shape[1])
        self.x_feature = st.slider('Variable en eje x', 1, n_features, 1)

        self.X = self.database.data[:, self.x_feature-1:self.x_feature]
        self.y = self.database.target
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=1234)
        self.sklearn_regr = P_sk(random_state=1234)
        self.sklearn_regr.fit(X_train, y_train)

        x1_min = self.X.min()
        x1_max = self.X.max()

        x_pred = np.linspace(x1_min, x1_max, 100).reshape([100, 1])
        y_pred = self.sklearn_regr.predict(x_pred)

        plt.figure(1, figsize=(12, 8))
        plt.scatter(self.X, self.y, edgecolors='k', cmap=plt.cm.Paired)
        plt.plot(x_pred, y_pred)
        return plt.gcf()
