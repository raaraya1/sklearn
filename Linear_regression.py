import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class linear_regression_st:
    def __init__(self, database, test_size=0.2):
        self.database = database
        self.test_size = test_size
        self.desc = r'''
        # **Linear Regression**


        **Predicción (aproximación)**
        $$
        \hat{y} = wx + b
        $$

        **Función de costos**

        $$
        Loss = MSE = \frac{1}{N} \sum_{i=1}^n (y_{i} - \hat{y_{i}})^2
        $$

        **Calculo del gradiente**


        $$
        \left[\begin{array}{ll}\frac{d_{loss}}{dw} \\ \frac{d_{loss}}{db} \end{array} \right] = \left[\begin{array}{ll} \frac{1}{N} \sum -2x_{i}(y_{i} - (wx_{i} + b)) \\ \frac{1}{N} \sum -2(y_{i} - (wx_{i} + b)) \end{array} \right]
        $$

        **Método del Descenso del Gradiente**

        - Inicializar los pesos ($w$) y el sesgo ($b$)
        - Iteramos
          - Calcular el gradiente
          - Actualizamos los parámetros (lr=learning rate)

        $$
        w = w - lr*dw
        $$

        $$
        b = b-lr*db
        $$

        - Terminamos de iterar
        '''

    def solve(self):
        self.X, self.y = self.database.data, self.database.target
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=1234)
        self.sklearn_regr = LinearRegression()
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
        self.sklearn_regr = LinearRegression()
        self.sklearn_regr.fit(X_train, y_train)

        x1_min = self.X.min()
        x1_max = self.X.max()

        x_pred = np.linspace(x1_min, x1_max, 100).reshape([100, 1])
        y_pred = self.sklearn_regr.predict(x_pred)

        plt.figure(1, figsize=(12, 8))
        plt.scatter(self.X, self.y, edgecolors='k', cmap=plt.cm.Paired)
        plt.plot(x_pred, y_pred)
        return plt.gcf()
