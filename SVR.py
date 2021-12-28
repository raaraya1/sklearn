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

        El objetivo es encontrar la función $f(x)$ que produzca el valor $y$ con una distancia no más lejana que $\epsilon$ para cada uno de los puntos de entrenamiento $x$.

        **Linear SVM Regression: Primal Formula**

        Supongamos que estamos trabajando con un set de datos X (multivariable) y con una variable dependiente y.

        Entonces la función lineal seria:

        $$
        f(X) = X^{T}\beta + b
        $$

        Luego, para asegurar que los parámetros $\beta$ sean lo más chicos (flat) posibles es que se busca minimizar:

        $$
        J(\beta) = \frac{1}{2}\beta^{T}\beta
        $$

        Restringido bajo las siguientes condiciones:

        $$
        |y_{n} - (X_{n}^{T}\beta + b)| \leq \epsilon \quad \forall n \in N
        $$

        Como es posible que no exista una función $f(x)$ que pueda satisfacer estas condiciones se introduce los términos $ℇ_{n}$ y $ℇ_{n}^{*}$ las cuales vienen a representar algo así como variables de holgura.

        Así, luego nuestra función objetivo cambia a:

        $$
        J(\beta) = \frac{1}{2}\beta^{T}\beta + C\sum_{n=1}^{N} (ℇ_{n} + ℇ_{n}^{*})
        $$

        Sujeto a:

        $$
        y_{n} - (X_{n}^{T} \beta +b) \leq \epsilon + ℇ_{n} \quad \forall n \in N
        $$

        $$
        (X_{n}^{T} \beta +b) - y_{n}\leq \epsilon + ℇ_{n}^{*} \quad \forall n \in N
        $$

        $$
        ℇ_{n}^{*} \geq 0 \quad \forall n \in N
        $$

        $$
        ℇ_{n} \geq 0 \quad \forall n \in N
        $$

        **Nota**: $C$ Son un conjunto de valores todos positivos que tiene por función penalizar las observaciones que se escapen del margen $\epsilon$

        **Nonlinear SVM Regression**

        En caso de que el problema no se pueda adaptar bien utilizando un modelo lineal, podemos adaptar todo este desarrollo cambiando el producto punto $X_{i}^{T}X_{j}$ por $G(X_{i}, X_{j})$.

        | Kernel Name | Kernel Function |
        |-------------|-----------------|
        |Linear (dot product)| $G(X_{i}, X_{j}) = X_{i}^{T}X_{j}$|
        |Gaussian|$G(X_{i}, X_{j}) = e^{-\lvert \rvert X_{i} - X_{j}^{2} \lvert \rvert}$|
        |Polynomial|$G(X_{i}, X_{j}) = (1 + X_{i}^{T}X_{j})^{q}$|

        **Nota:** $q$ es el grado del polinomio

        **Fuente**: https://www.mathworks.com/help/stats/understanding-support-vector-machine-regression.html

'''

    def params(self):
        self.selected_kernel = st.selectbox('Tipo de kernel:', options=['linear', 'poly', 'rbf', 'sigmoid'])
        if self.selected_kernel == 'poly': self.degree = st.slider('Grados del polinomio', 1, 6, 3)
        min = float(np.min([0, np.min(self.database.target)]))/2
        max = float(np.max(self.database.target))/2
        mean = float(np.mean(self.database.target))/2
        self.C = st.slider('Parametro de penalizacion C:', 1.0, 4*max, 4*mean)
        self.epsilon = st.slider('Epsilon: ', min, max, mean)

    def solve(self):
        self.X, self.y = self.database.data, self.database.target
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=1234)
        if self.selected_kernel == 'poly': self.sklearn_regr = SVR(kernel=self.selected_kernel,
                                                                    degree=self.degree,
                                                                    C=self.C,
                                                                    epsilon=self.epsilon)
        else: self.sklearn_regr = SVR(kernel=self.selected_kernel,
                                        C=self.C,
                                        epsilon=self.epsilon)
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
        if self.selected_kernel == 'poly': self.sklearn_regr = SVR(kernel=self.selected_kernel,
                                                                    degree=self.degree,
                                                                    C=self.C,
                                                                    epsilon=self.epsilon)
        else: self.sklearn_regr = SVR(kernel=self.selected_kernel,
                                        C=self.C,
                                        epsilon=self.epsilon)
        self.sklearn_regr.fit(X_train, y_train)

        x1_min = self.X.min()
        x1_max = self.X.max()

        x_pred = np.linspace(x1_min, x1_max, 100).reshape([100, 1])
        y_pred = self.sklearn_regr.predict(x_pred)
        y_pred_up = [i+self.epsilon for i in y_pred]
        y_pred_down = [i-self.epsilon for i in y_pred]


        plt.figure(1, figsize=(12, 8))
        plt.scatter(self.X, self.y, edgecolors='k', cmap=plt.cm.Paired)
        plt.plot(x_pred, y_pred, color='red') # linea de prediccion
        plt.plot(x_pred, y_pred_up, linestyle='--', color='green')
        plt.plot(x_pred, y_pred_down, linestyle='--', color='green')
        return plt.gcf()
