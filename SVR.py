import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

class SVR_st:
    def __init__(self, database, test_size=0.2):
        self.database = database
        self.test_size = test_size
        self.desc = r'''
        # **SVR (Support Vector Regression)**

        En construccion.
        '''

    def params(self):
        st.write('En construccion')

    def solve(self):
        X, y = self.database.data, self.database.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
        sklearn_regr = SVR()
        sklearn_regr.fit(X_train, y_train)
        y_pred = sklearn_regr.predict(X_test)
        acc = mean_squared_error(y_pred, y_test)
        st.metric('MSE (Mean Square Error)', value=f'{np.round(acc, 2)}')
