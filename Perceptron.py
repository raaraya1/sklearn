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

        Este es el modelo mas sencillo y que sirve de introduccion a los modelos de redes neuronales. En particular, su funcionamiento es bastante similar al modelo de regresion linea. con la diferencia de que ocupa una funcion de activacion en la salida (**funcion no lineal**).

        **Modelo Lineal**

        $$
        f(w, b) = w^{t}x + b
        $$

        **Funcion de Activacion**

        $$
        z(x) \in (0, 1) \quad si \quad x \geq 0
        $$

        **Aproximacion (prediccion)**

        $$
        \hat{y} = z(w^{t}x + b)
        $$

        **Reglas de actualización (aqui se encuentra incluido el bias)**

        $$
        w = w + \Delta w = w + lr(y_{i} - \hat{y_{i}})x_{i}
        $$
        '''

    def solve(self):
        X, y = self.database.data, self.database.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=1234)
        self.sklearn_regr = P_sk(random_state=1234)
        self.sklearn_regr.fit(X_train, y_train)
        y_pred = self.sklearn_regr.predict(X_test)
        acc = mean_squared_error(y_pred, y_test)
        st.metric('MSE (Mean Square Error)', value=f'{np.round(acc, 2)}')
