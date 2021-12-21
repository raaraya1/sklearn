import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class linear_regression_st:
    def __init__(self, database, test_size=0.2):
        self.database = database
        self.test_size = test_size
        self.desc = r'''
        # **Linear Regression**


        **Prediccion (aproximacion)**
        $$
        \hat{y} = wx + b
        $$

        **Funcion de costos**

        $$
        Loss = MSE = \frac{1}{N} \sum_{i=1}^n (y_{i} - \hat{y_{i}})^2
        $$

        **Calculo del gradiente**


        $$
        \left[\begin{array}{ll}\frac{d_{loss}}{dw} \\ \frac{d_{loss}}{db} \end{array} \right] = \left[\begin{array}{ll} \frac{1}{N} \sum -2x_{i}(y_{i} - (wx_{i} + b)) \\ \frac{1}{N} \sum -2(y_{i} - (wx_{i} + b)) \end{array} \right]
        $$

        **Metodo del Descenso del Gradiente**

        - Inicializar los pesos ($w$) y el sesgo ($b$)
        - Iteramos
          - Calcular el gradiente
          - Actualizamos los parametros (lr=learning rate)

        $$
        w = w - lr*dw
        $$

        $$
        b = b-lr*db
        $$

        - Terminamos de iterar
        '''

    def solve(self):
        X, y = self.database.data, self.database.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
        sklearn_regr = LinearRegression()
        sklearn_regr.fit(X_train, y_train)
        y_pred = sklearn_regr.predict(X_test)
        acc = mean_squared_error(y_pred, y_test)
        st.metric('MSE (Mean Square Error)', value=f'{np.round(acc, 2)}')
