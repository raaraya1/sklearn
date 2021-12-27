import streamlit as st
from sklearn import datasets
from KNN import KNN_st
from SVC import SVC_st
from Logit import Logit_st
from Decision_tree import Decision_tree_st
from Random_forest import random_forest_st
from Naive_bayes import naive_bayes_st
from Ada_boost import ada_boost_st
from Linear_regression import linear_regression_st
from SVR import SVR_st
from Perceptron import perceptron_st
from k_mean_clustering import k_mean_clustering_st, plot
from PCA import PCA_st
from ICA import ICA_st
from Agglomerative_clustering import agglomerative_clustering_st
from LDA import LDA_st

st.write('''
# **Machine Learning**

Esta DEMO tiene por objetivo mostrar de manera didáctica algunos de los algoritmos
que más frecuentemente se utilizan en **Machine Learning**. Así, la biblioteca de
`sklearn` la podríamos separar en 2 grandes grupos, los cuales se encuentran demarcados
en función del objetivo que se pretende conseguir.

- **Supervised Learning**
- **Unsupervised Learning**

''')

task = st.sidebar.selectbox('Tipo de algoritmo:', options=['Supervised Learning', 'Unsupervised Learning'])

# ----------------------------------------Supervised Learning-------------------------------
if task == 'Supervised Learning':
    st.write('''
    #
    ## **Supervised Learning**

    **Supervised learning** consiste en aprender sobre la relación entre dos conjuntos de datos:
    - Las observaciones (X)
    - La variable externa (y), sobre la cual generalmente se pretende predecir (target o label)

    Todos los estimadores de la biblioteca de sklearn tiene implementado el método
    `fit(X, y)` para ajustar el algoritmo a los datos y el método `predict(X)` para
    etiquetar las observaciones X.

    **Classification and regression**

    Si la tarea sobre la predicción consiste en clasificar las observaciones en
    un numero finito de "etiquetas" (en otras palabras, nombrar el objeto mostrado),
    entonces se dice que estamos hablando de una tarea de **Clasificación**.
    Por otro lado, si la predicción es sobre una variable continua, entonces estamos
    hablando de una tarea de **Regresión**.
''')

    type = st.sidebar.radio('Objetivo del algoritmo:', options=['Classification', 'Regression'])
    if type == 'Classification':
        dataset_selected = None

        # Seleccionamos la base de datos (estas son para clcasificacion)
        with st.expander('Base de datos'):
            class_sets = ['iris', 'digits', 'breast cancer', 'wine']
            dataset_name = st.selectbox('Escoja una base de datos', options=class_sets)
            if dataset_name == 'iris':
                dataset_selected = datasets.load_iris()
                st.write(f'{dataset_selected.DESCR}')
            elif dataset_name == 'digits':
                dataset_selected = datasets.load_digits()
                st.write(f'{dataset_selected.DESCR}')
            elif dataset_name == 'breast cancer':
                dataset_selected = datasets.load_breast_cancer()
                st.write(f'{dataset_selected.DESCR}')
            elif dataset_name == 'wine':
                dataset_selected = datasets.load_wine()
                st.write(f'{dataset_selected.DESCR}')


        alg_selected = st.sidebar.selectbox('Algoritmo:', ['SVC (Support Vector Classification)',
                                            'KNN (K Nearest Neighborns)',
                                            'Logistic Regression',
                                            'Decision Tree',
                                            'Random Forest',
                                            'Naive Bayes',
                                            'Ada Boost'])
        # seleccionar el algoritmo
        if alg_selected == 'KNN (K Nearest Neighborns)': algorithm = KNN_st(dataset_selected)
        elif alg_selected == 'SVC (Support Vector Classification)': algorithm = SVC_st(dataset_selected)
        elif alg_selected == 'Logistic Regression': algorithm = Logit_st(dataset_selected)
        elif alg_selected == 'Decision Tree': algorithm = Decision_tree_st(dataset_selected)
        elif alg_selected == 'Random Forest': algorithm = random_forest_st(dataset_selected)
        elif alg_selected == 'Naive Bayes': algorithm = naive_bayes_st(dataset_selected)
        elif alg_selected == 'Ada Boost': algorithm = ada_boost_st(dataset_selected)


        with st.expander('Explicacion del algoritmo'):
            if alg_selected == 'KNN (K Nearest Neighborns)': algorithm.desc
            elif alg_selected == 'SVC (Support Vector Classification)': algorithm.desc
            elif alg_selected == 'Logistic Regression': algorithm.desc
            elif alg_selected == 'Decision Tree': algorithm.desc
            elif alg_selected == 'Random Forest': algorithm.desc
            elif alg_selected == 'Naive Bayes': algorithm.desc
            elif alg_selected == 'Ada Boost': algorithm.desc


        with st.expander('Ajustes de parametros'):
            if alg_selected == 'KNN (K Nearest Neighborns)': algorithm.params()
            elif alg_selected == 'SVC (Support Vector Classification)': algorithm.params()
            elif alg_selected == 'Logistic Regression': algorithm.params()
            elif alg_selected == 'Decision Tree': algorithm.params()
            elif alg_selected == 'Random Forest': algorithm.params()
            elif alg_selected == 'Naive Bayes': pass
            elif alg_selected == 'Ada Boost': algorithm.params()

        with st.expander('Resultados'):
            if alg_selected == 'KNN (K Nearest Neighborns)': algorithm.solve()
            elif alg_selected == 'SVC (Support Vector Classification)': algorithm.solve()
            elif alg_selected == 'Logistic Regression': algorithm.solve()
            elif alg_selected == 'Decision Tree': algorithm.solve()
            elif alg_selected == 'Random Forest': algorithm.solve()
            elif alg_selected == 'Naive Bayes': algorithm.solve()
            elif alg_selected == 'Ada Boost': algorithm.solve()

        with st.expander('Visualizacion'):
            c = st.container()
            if alg_selected == 'KNN (K Nearest Neighborns)': c.pyplot(algorithm.visualization())
            elif alg_selected == 'SVC (Support Vector Classification)': c.pyplot(algorithm.visualization())
            elif alg_selected == 'Logistic Regression': c.pyplot(algorithm.visualization())
            elif alg_selected == 'Decision Tree': c.pyplot(algorithm.visualization())
            elif alg_selected == 'Random Forest': c.pyplot(algorithm.visualization())
            elif alg_selected == 'Naive Bayes': c.pyplot(algorithm.visualization())
            elif alg_selected == 'Ada Boost': c.pyplot(algorithm.visualization())



    elif type == 'Regression':
        dataset_selected = None

        # Seleccionamos la base de datos (estas son para Regresiones)
        with st.expander('Base de datos'):
            class_sets = ['diabetes', 'boston']
            dataset_name = st.selectbox('Escoja una base de datos', options=class_sets)
            if dataset_name == 'diabetes':
                dataset_selected = datasets.load_diabetes()
                st.write(f'{dataset_selected.DESCR}')
            elif dataset_name == 'boston':
                dataset_selected = datasets.load_boston()
                st.write(f'{dataset_selected.DESCR}')


        alg_selected = st.sidebar.selectbox('Algoritmo:', ['Linear Regression',
                                            'SVR (Support Vector Regression)',
                                                'Perceptron'])
        # seleccionar el algoritmo
        if alg_selected == 'Linear Regression': algorithm = linear_regression_st(dataset_selected)
        elif alg_selected == 'SVR (Support Vector Regression)': algorithm = SVR_st(dataset_selected)
        elif alg_selected == 'Perceptron': algorithm = perceptron_st(dataset_selected)

        with st.expander('Explicacion del algoritmo'):
            if alg_selected == 'Linear Regression': algorithm.desc
            elif alg_selected == 'SVR (Support Vector Regression)': algorithm.desc
            elif alg_selected == 'Perceptron': algorithm.desc

        with st.expander('Ajustes de parametros'):
            if alg_selected == 'Linear Regression': pass
            elif alg_selected == 'SVR (Support Vector Regression)': algorithm.params()
            elif alg_selected == 'Perceptron': pass

        with st.expander('Resultados'):
            if alg_selected == 'Linear Regression': algorithm.solve()
            elif alg_selected == 'SVR (Support Vector Regression)': algorithm.solve()
            elif alg_selected == 'Perceptron': algorithm.solve()

        with st.expander('Visualización'):
            c = st.container()
            if alg_selected == 'Linear Regression': c.pyplot(algorithm.visualization())
            elif alg_selected == 'SVR (Support Vector Regression)': c.pyplot(algorithm.visualization())
            elif alg_selected == 'Perceptron': c.pyplot(algorithm.visualization())


# ------------------------------------Unsupervised learning-----------------------------------

elif task == 'Unsupervised Learning':
    st.write('''
    #
    ## **Unsupervised Learning**

    **Unsupervised learning**: Para este tipo de aprendizaje los datos no vienen
    con un objetivo (**target**). De esta manera, lo que se busca es descubrir los
    grupos con mayores características similares (**clustering**) o determinar
    la distribución de los datos en el espacio (luego si esta distribución se
    encuentra en muchas dimensiones, la podemos reducir a 2 o 3 con fin de poder
    visualizar los datos)

    ''')
    alg_selected = st.sidebar.selectbox('Algoritmo:', ['K-means Clustering',
                                        'Agglomerative Clustering',
                                        'PCA (Principal Component Analysis)',
                                        'ICA (Independent Component Analysis)',
                                        'LDA (Linear Discrimination Analysis)'])
    dataset_selected = None

    # Seleccionamos la base de datos (todas las bases sirven)
    with st.expander('Base de datos'):
        class_sets = ['iris', 'digits', 'breast cancer', 'diabetes', 'wine', 'boston']
        dataset_name = st.selectbox('Escoja una base de datos', options=class_sets)
        if dataset_name == 'iris':
            dataset_selected = datasets.load_iris()
            st.write(f'{dataset_selected.DESCR}')
        elif dataset_name == 'digits':
            dataset_selected = datasets.load_digits()
            st.write(f'{dataset_selected.DESCR}')
        elif dataset_name == 'breast cancer':
            dataset_selected = datasets.load_breast_cancer()
            st.write(f'{dataset_selected.DESCR}')
        elif dataset_name == 'diabetes':
            dataset_selected = datasets.load_diabetes()
            st.write(f'{dataset_selected.DESCR}')
        elif dataset_name == 'wine':
            dataset_selected = datasets.load_wine()
            st.write(f'{dataset_selected.DESCR}')
        elif dataset_name == 'boston':
            dataset_selected = datasets.load_boston()
            st.write(f'{dataset_selected.DESCR}')



    # seleccionar el algoritmo
    if alg_selected == 'K-means Clustering': algorithm = k_mean_clustering_st(dataset_selected)
    elif alg_selected == 'PCA (Principal Component Analysis)': algorithm = PCA_st(dataset_selected)
    elif alg_selected == 'ICA (Independent Component Analysis)': algorithm = ICA_st(dataset_selected)
    elif alg_selected == 'Agglomerative Clustering': algorithm = agglomerative_clustering_st(dataset_selected)
    elif alg_selected == 'LDA (Linear Discrimination Analysis)': algorithm = LDA_st(dataset_selected)

    with st.expander('Explicacion del algoritmo'):
        if alg_selected == 'K-means Clustering': algorithm.desc
        elif alg_selected == 'PCA (Principal Component Analysis)': algorithm.desc
        elif alg_selected == 'ICA (Independent Component Analysis)': algorithm.desc
        elif alg_selected == 'Agglomerative Clustering': algorithm.desc
        elif alg_selected == 'LDA (Linear Discrimination Analysis)': algorithm.desc

    with st.expander('Ajustes de parametros'):
        if alg_selected == 'K-means Clustering': algorithm.params()
        elif alg_selected == 'PCA (Principal Component Analysis)': algorithm.params()
        elif alg_selected == 'ICA (Independent Component Analysis)': algorithm.params()
        elif alg_selected == 'Agglomerative Clustering': algorithm.params()
        elif alg_selected == 'LDA (Linear Discrimination Analysis)': algorithm.params()

    with st.expander('Resultados'):
        c = st.container()
        if alg_selected == 'K-means Clustering': c.pyplot(algorithm.solve())
        elif alg_selected == 'PCA (Principal Component Analysis)': c.pyplot(algorithm.solve())
        elif alg_selected == 'ICA (Independent Component Analysis)': c.pyplot(algorithm.solve())
        elif alg_selected == 'Agglomerative Clustering': c.pyplot(algorithm.solve())
        elif alg_selected == 'LDA (Linear Discrimination Analysis)' and (dataset_selected.DESCR).split()[1] not in ['_diabetes_dataset:', '_boston_dataset:']:
            c.pyplot(algorithm.solve())
        else:
            st.write('''
            **Nota:** LDA solo puede resolver problemas de clasificación, ya que require de las
            **etiquetas** de las observaciones para funcionar''')
