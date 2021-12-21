import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


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
        X, y = self.database.data, self.database.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
        sklearn_clf = GaussianNB()
        sklearn_clf.fit(X_train, y_train)
        y_pred = sklearn_clf.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        st.metric('Acierto', value=f'{np.round(acc, 2)*100}%')
