import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



class SVC_st:
    def __init__(self, database, test_size=0.2):
        self.database = database
        self.test_size = test_size
        self.desc = r'''
                        # **Support Vector Machine**

        Este algoritmo tiene por objetivo la búsqueda de un hiperplano que segregue los datos atendiendo a estas dos condiciones:

        $$
        wx - b = 0
        $$

        $$
        max \quad \frac{2}{||w||}
        $$

        **Linear model (2 categorías (1 y -1))**

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

        **Función de costos (loss)**

        $$
        loss = λ||w||^2 + \frac{1}{n} \sum_{i=1}^{n} max(0, 1-y_{i}(wx_{i}-b))
        $$

        De esta manera las **derivadas** en función de los parámetros siguen las siguientes reglas:

        - si $y_{i}(xw - b) \geq 1$:

        $$
        \left[\begin{array}{ll} \frac{d_{loss}}{d_{w_{k}}} \\ \frac{d_{loss}}{db} \end{array} \right] = \left [\begin{array}{ll} 2 \lambda w_{k} \\ 0 \end{array} \right]
        $$

        - si $y_{i}(xw - b) < 1$:

        $$
        \left[\begin{array}{ll}\frac{d_{loss}}{d_{w_{k}}} \\ \frac{d_{loss}}{db} \end{array} \right] = \left[\begin{array}{ll} 2\lambda w_{k} - y_{i} \cdot x_{i} \\ y_{i} \end{array} \right]
        $$

        **Reglas de actualización (Gradient Descent)**

        - Inicializar parámetros
        - Iterar
         - Calcular loss
         - Calcular gradiente
         - Actualizar parámetros

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
        self.X, self.y = self.database.data, self.database.target
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=1234)
        self.sklearn_clf = svm.SVC(kernel=self.kernel, gamma=self.gamma, random_state=1234)
        self.sklearn_clf.fit(X_train, y_train)
        y_pred = self.sklearn_clf.predict(X_test)
        acc = accuracy_score(y_pred, y_test)

        c1, c2 = st.columns([4, 1])
        c2.metric('Acierto', value=f'{np.round(acc, 2)*100}%')
        df = pd.DataFrame(confusion_matrix(y_pred, y_test))
        labels = self.database.target_names
        df.columns = labels
        df.index = labels
        c1.write('**Confusion Matrix**')
        c1.dataframe(df)

    def visualization(self):
        n_features = int(self.database.data.shape[1])
        self.x_feature = st.slider('Variables en eje x', 1, n_features, 1)
        self.y_feature = st.slider('Variables en eje y', 1, n_features, 2)

        self.X = np.c_[self.database.data[:, self.x_feature-1:self.x_feature], self.database.data[:, self.y_feature-1:self.y_feature]]
        self.y = self.database.target
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=1234)
        self.sklearn_clf = svm.SVC(kernel=self.kernel, gamma=self.gamma, random_state=1234)
        self.sklearn_clf.fit(X_train, y_train)

        x1_min, x1_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        x2_min, x2_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        h = 0.02 # Salto que vamos dando
        x1_i = np.arange(x1_min, x1_max, h)
        x2_i = np.arange(x2_min, x2_max, h)
        x1_x1, x2_x2 = np.meshgrid(x1_i, x2_i)
        y_pred = self.sklearn_clf.predict(np.c_[x1_x1.ravel(), x2_x2.ravel()])
        y_pred = y_pred.reshape(x1_x1.shape)

        plt.figure(1, figsize=(12, 8))
        plt.pcolormesh(x1_x1, x2_x2, y_pred, cmap=plt.cm.Paired)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlim(x1_x1.min(), x1_x1.max())
        plt.ylim(x2_x2.min(), x2_x2.max())
        return plt.gcf()
