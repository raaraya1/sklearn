import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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
        self.X, self.y = self.database.data, self.database.target
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=1234)
        self.sklearn_regr = SVR()
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
        self.sklearn_regr = SVR()
        self.sklearn_regr.fit(X_train, y_train)

        x1_min = self.X.min()
        x1_max = self.X.max()

        x_pred = np.linspace(x1_min, x1_max, 100).reshape([100, 1])
        y_pred = self.sklearn_regr.predict(x_pred)

        plt.figure(1, figsize=(12, 8))
        plt.scatter(self.X, self.y, edgecolors='k', cmap=plt.cm.Paired)
        plt.plot(x_pred, y_pred)
        return plt.gcf()
