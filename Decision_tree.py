import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class Decision_tree_st:
    def __init__(self, database, test_size=0.2):
        self.database = database
        self.test_size = test_size
        self.desc = r'''
        # **Decision Tree**

        **Entropy**

        $$
        E = - \sum p(X) \cdot log_{2}(p(X))
        $$

        $$
        p(X) = \frac{len(x)}{n}
        $$

        **Ganancia de informacion**

        $$
        IG = E(parent) - [weight \quad average] \cdot E(children)
        $$

        **Metodo (para construir el arbol)**

        - Se comienza desde el primer nodo y para cada se selecciona la mejor separacion en base a la ganancia de informacion.
        - De la ganancia de informacion mas alta se rescata la variable y el limite.
        - Luego se aplica la segmentacion a cada nodo, en base a la variable y limite encontrado.
        - Se itera con estos pasos hasta cumplirse algun criterio
         - **maximium depth**: cantidad de nodos maximos al final
         - **minimum samples**: cantidad minima de elementos que puede tener los nodos
         - **no more class distribution**: No existen mas elementos para segmentar

        **Aproximacion (prediccion)**

        - Se sigue las segmentaciones en el orden del arbol (de arriba a abajo)
        - Cuando se llega a un nodo al final del arbol se predice segun el valor mas comun en esa muestra.


        '''
        self.max_depth = 100
        self.min_samples_split = 2
        self.stop_criterion = 'max_depth'

    def params(self):
        self.stop_criterion = st.radio('Criterio de termino:', options=['max_depth', 'min_samples_split'])
        if self.stop_criterion == 'max_depth': self.max_depth = st.slider('Valor max deph:', 1, 100, 10)
        elif self.stop_criterion == 'min_samples_split': self.min_samples_split = st.slider('Valor min_samples_split:', 2, 1000, 5)


    def solve(self):
        X, y = self.database.data, self.database.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
        if self.stop_criterion == 'max_depth': sklearn_clf = DecisionTreeClassifier(max_depth=self.max_depth)
        elif self.stop_criterion == 'min_samples_split': sklearn_clf = DecisionTreeClassifier(min_samples_split=self.min_samples_split)

        sklearn_clf.fit(X_train, y_train)
        y_pred = sklearn_clf.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        st.metric('Acierto', value=f'{np.round(acc, 2)*100}%')
