import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.metrics import accuracy_score


class ada_boost_st:
    def __init__(self, database, test_size=0.2):
        self.database = database
        self.test_size = test_size
        self.desc = r'''
        # **AdaBoost**

        Este algoritmo se basa en ir agrupando otros algoritmos de clasificacion, para que en conjunto generen una prediccion.

        Asimismo y, a diferencia del algoritmo de Random Forest, es que el **voto** de cada estimador no valen lo mismo, es decir, existe un grado de importancia (**weight**) entre los estimadores que siendo estos ponderados por sus votos es que generan la prediccion del algoritmo.

        **Weak Learner (Decision Stump)**

        Es un algoritmo que sencillamente clasifica los datos segun un limite (similar a uno de los pasos del algoritmo de Decision Tree)

        **Error**

        - Primera itereacion

        $$
        ϵ_{1} = \frac{desaceirtos}{N}
        $$

        - A partir de la segunda iteracion

        $$
        ϵ_{t} = \sum weights
        $$

        Nota: Si el error es mayor a 0.5, se itercambia la clasificacion y se calcula el $error = 1 - error$

        **Weights**

        - Al inicio
        $$
        w_{0} = \frac{1}{N} para cada muestra
        $$

        - Luego

        $$
        w = \frac{w \cdot e^{- αyh(X)}}{\sum w}
        $$

        **Performance**

        $$
        \alpha = 0.5 \cdot log(\frac{1-ϵ_{t}}{ϵ_{t}})
        $$

        **Prediction**

        $$
        y = sign(\sum_{t}^{T} α_{t} \cdot h(X))
        $$

        **Training**

        Se inicializan los pesos de cada mustra en $\frac{1}{N}$

        - Entrenamos a un clasificador debil (se busca la mejor variable y limite para segmentar)
        - Calculamos el error $ϵ_{t} = \sum_{desaciertos} weights$
         - Cambiar el error y la polaridad si este es mayor a 0.5
        - Calcular $ \alpha = 0.5 \cdot log(\frac{1 - \epsilon_{t}}{ϵ_{t}})$
        - Actualizar los pesos: $w = \frac{w \cdot e^{- αh(X)}}{Z}$
        '''
        self.n_clf = 5

    def params(self):
        self.n_clf = st.slider('Numero de estimadores',
                                    min_value=1,
                                    max_value=15,
                                    value=5)

    def solve(self):
        X, y = self.database.data, self.database.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
        sklearn_clf = ABC(n_estimators=self.n_clf)
        sklearn_clf.fit(X_train, y_train)
        y_pred = sklearn_clf.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        st.metric('Acierto', value=f'{np.round(acc, 2)*100}%')
