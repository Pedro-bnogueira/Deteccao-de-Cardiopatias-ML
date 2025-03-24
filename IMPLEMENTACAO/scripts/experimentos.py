# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
# INTEGRANTES:
#
# Nome: Rafael Mori Pinheiro
# RA: 813851
#
# Nome: Pedro Enrico Barchi Nogueira
# RA: 813099
#
# ################################################################

# Arquivo com todas as funções e códigos referentes aos experimentos

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

"""
Cria um dicionário contendo os modelos que serão testados no experimento

Modelos incluídos:
- k-NN
- Naive Bayes 
- Regressão Logística
- Redes Neurais Artificiais (MLP)
- SVM
- Random Forest
- Gradient Boosting
- Ada Boost

Retorna:
- modelos (dict): Dicionário onde as chaves são os nomes dos modelos e os valores são as instâncias dos classificadores
"""
def criar_modelos():
    modelos = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'NaiveBayes': GaussianNB(),
        'LogisticRegression': LogisticRegression(max_iter=500, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(100,),
                            max_iter=1000, 
                            random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42, n_iter_no_change=5, validation_fraction=0.1, tol=1e-4),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    }
    return modelos

"""
Retorna um dicionário de hiperparâmetros específico para cada modelo

Parâmetros:
- modelo (sklearn estimator): Instância do modelo de machine learning

Retorna:
- param_grid (dict ou None): Dicionário com a grade de busca de hiperparâmetros para o modelo
"""
def get_param_grid(modelo):
    if isinstance(modelo, KNeighborsClassifier):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
        return param_grid

    elif isinstance(modelo, GaussianNB):
        param_grid = {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
        return param_grid

    elif isinstance(modelo, LogisticRegression):
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'],  
            'penalty': ['l2']                 
        }
        return param_grid

    elif isinstance(modelo, MLPClassifier):
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (100,100)],
            'alpha': [1e-4, 1e-3, 1e-2],
            'learning_rate_init': [0.001, 0.01]
        }
        return param_grid

    elif isinstance(modelo, SVC):
        param_grid = {
            'C': [1.0],
            'kernel': ['linear']
        }
        return param_grid

    elif isinstance(modelo, RandomForestClassifier):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [2, 4],              
            'max_features': ['sqrt', 0.8]             
        }
        return param_grid

    elif isinstance(modelo, GradientBoostingClassifier):
        param_grid = {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 1.0],
            'min_samples_leaf': [1, 2, 5]
        }
       
        return param_grid

    elif isinstance(modelo, AdaBoostClassifier):
        param_grid = {
            'n_estimators': [50, 100], 
            'learning_rate': [0.01, 0.1, 1]
        }
        return param_grid

    return None

"""
Treina o modelo com os dados de treinamento

Parâmetros:
- modelo (sklearn estimator): Instância do modelo de machine learning
- X_train (pd.DataFrame ou np.array): Features do conjunto de treinamento
- y_train (pd.Series ou np.array): Labels do conjunto de treinamento

Retorna:
- modelo treinado (sklearn estimator): Modelo ajustado aos dados de treinamento 
"""
def treinar_modelo(modelo, X_train, y_train):
    # Obter a grade de parâmetros adequada ao modelo
    param_grid = get_param_grid(modelo)

    if param_grid is not None:
        grid = GridSearchCV(
            estimator=modelo,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=5,                
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        print(f"[INFO] {modelo.__class__.__name__} - Melhores hiperparâmetros: {grid.best_params_}")
        return best_model
    else:
        modelo.fit(X_train, y_train)
        return modelo