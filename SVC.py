import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


class SVC_st:
    def __init__(self, database, test_size=0.2):
        self.database = database
        self.test_size = test_size
        self.desc = r'''
                        # **Support Vector Machine**

        Este algortimo tiene por objetivo la busqueda de un hiperplano que segregue los datos atendiendo a estas dos condiciones:

        $$
        wx - b = 0
        $$

        $$
        max \quad \frac{2}{||w||}
        $$

        **Linear model (2 categorias (1 y -1))**

        $$
        wx - b = 0
        $$

        $$
        wx_{i} - b \geq 1 \quad si \quad y_{i} = 1
        $$

        $$
        wx_{i} - b \leq 1 \quad si \quad y_{i} = -1
        $$

        **Estas 3 ecuaciones se resumen en la siguiente:**

        $$
        y_{i}(wx_{i} - b) \geq 1
        $$

        **Funcion de costos (loss)**

        $$
        loss = λ||w||^2 + \frac{1}{n} \sum_{i=1}^{n} max(0, 1-y_{i}(wx_{i}-b))
        $$

        De esta manera las **derivadas** en funcion de los parametros siguen las siguientes reglas:

        - si $y_{i}(xw - b) \geq 1$:

        $$
        \left[\begin{array}{ll} \frac{d_{loss}}{d_{w_{k}}} \\ \frac{d_{loss}}{db} \end{array} \right] = \left [\begin{array}{ll} 2 \lambda w_{k} \\ 0 \end{array} \right]
        $$

        - si $y_{i}(xw - b) < 1$:

        $$
        \left[\begin{array}{ll}\frac{d_{loss}}{d_{w_{k}}} \\ \frac{d_{loss}}{db} \end{array} \right] = \left[\begin{array}{ll} 2\lambda w_{k} - y_{i} \cdot x_{i} \\ y_{i} \end{array} \right]
        $$

        **Reglas de actualizacion (Gradient Descent)**

        - Inicializar parametros
        - Iterar
         - Calcular loss
         - Calcular gradiente
         - Actualizar parametros

         $$
        w = w - lr \cdot dw
         $$

         $$
        b = b - lr \cdot db
         $$

        - Terminar de iterar
        '''
        self.kernel = 'linear'
        self.gamma = 2
        self.degree = 3

    def params(self):
        tipo = st.selectbox('Tipo de kernel', options=['linear',
                                                        'poly',
                                                        'rbf'])
        self.kernel = tipo
        self.gamma = st.slider('Parametro gamma', 1, 10, 2)
        if tipo == 'poly': self.degree = st.slider('Cantidad de grados del polinomio', 1, 10, 3)


    def solve(self):
        X, y = self.database.data, self.database.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
        sklearn_clf = svm.SVC(kernel=self.kernel, gamma=self.gamma)
        sklearn_clf.fit(X_train, y_train)
        y_pred = sklearn_clf.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        st.metric('Acierto', value=f'{np.round(acc, 2)*100}%')
