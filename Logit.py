import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR_sk
from sklearn.metrics import accuracy_score


class Logit_st:
    def __init__(self, database, test_size=0.2):
        self.database = database
        self.test_size = test_size
        self.desc = r'''
        # **Logistic Regresion**

        **Prediccion (Aproximacion)**

        $$
        z = wx + b
        $$

        $$
        \hat{y} = \frac{1}{1+e^{-z}}
        $$

        **Funcion de perdida (cross entropy)**

        $$
        loss = \frac{1}{N} \sum_{i=1}^{n} [y^{i}log(\hat{y(x^{i})}) + (1-y^{i})log(1 - \hat{y(x^{i})})]
        $$

        **Gradientes**

        $$
        \left[\begin{array}{11} \frac{d_{loss}}{dw} \\ \frac{d_{loss}}{db} \end{array}\right] = \left[\begin{array}{11} \frac{1}{N} \sum 2x_{i}(\hat{y} - y_{i}) \\ \frac{1}{N} \sum 2(\hat{y} - y_{i}) \end{array}\right]
        $$

        **Metodo de Gradient Descent**
        - Iniciar parametros
        - Iterar
         - Calcular el error (loss)
         - Actualizar los pesos ($lr$=learning rate)

         $$
        w = w - lr*dw
         $$

         $$
         b = b - lr*db
         $$

        - Terminar de iterar
        '''


    def solve(self):
        X, y = self.database.data, self.database.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
        sklearn_clf = LR_sk(max_iter=1000)
        sklearn_clf.fit(X_train, y_train)
        y_pred = sklearn_clf.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        st.metric('Acierto', value=f'{np.round(acc, 2)*100}%')
