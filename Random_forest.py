import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import accuracy_score


class random_forest_st:
    def __init__(self, database, test_size=0.2):
        self.database = database
        self.test_size = test_size
        self.desc = '''
        # **Random Forest**

        Este algoritmo se contruye en base al algoritmo de **Decision Tree**. Asi, lo que se hace es:

        - Definir cantidad de estimadores (**Decision Tree**)
        - Cada estimador entrenarlo con una muestra del set de datos de entrenamiento, variando asi la cantidad de variables y la cantidad de datos con la cual se entrenan estos estimadores.
        - Luego, para generar la prediccion de algoritmo, lo que se hace es consultar a cada estimador su prediccion y "**de manera democratica**" se escoje la opción mas "**votada**"
        '''
        self.n_trees = 100
        self.min_samples_split = 2
        self.max_depth = 100
        self.n_feats = None
        self.stop_criterion = 'max_depth'


    def params(self):
        self.stop_criterion = st.radio('Criterio de termino:', options=['max_depth', 'min_samples_split'])
        if self.stop_criterion == 'max_depth': self.max_depth = st.slider('Valor max deph:', 1, 100, 10)
        elif self.stop_criterion == 'min_samples_split': self.min_samples_split = st.slider('Valor min_samples_split:', 2, 1000, 5)
        self.n_trees = st.slider('Cantidad de estimadores: ', 1, 100, 3)
        self.n_feats = st.slider('Fraccion de categorias para contruir los estimadores: ', 0.0, 1.0, 0.5)

    def solve(self):
        X, y = self.database.data, self.database.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
        if self.stop_criterion == 'max_depth': sklearn_clf = rf(n_estimators=self.n_trees,
                                                                max_depth=self.max_depth,
                                                                max_features=self.n_feats)
        elif self.stop_criterion == 'min_samples_split': sklearn_clf = rf(n_estimators=self.n_trees,
                                                                min_samples_split=self.min_samples_split, 
                                                                max_features=self.n_feats)

        sklearn_clf = rf(n_estimators=self.n_trees)
        sklearn_clf.fit(X_train, y_train)
        y_pred = sklearn_clf.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        st.metric('Acierto', value=f'{np.round(acc, 2)*100}%')
