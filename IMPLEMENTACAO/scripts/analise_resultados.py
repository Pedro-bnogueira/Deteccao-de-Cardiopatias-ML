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

# Arquivo com todas as funções e códigos referentes à análise dos resultados

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, learning_curve
import joblib

"""
Avalia o modelo utilizando métricas como acurácia, precisão, recall, F1-score e auc-roc

Parâmetros:
- modelo: Modelo treinado 
- X: Features de entrada
- y: Labels reais

Retorna:
- dict: Dicionário com as métricas calculadas
"""
def avaliar_modelo(modelo, X, y):
    # Obter as previsões
    y_pred = modelo.predict(X)

    # Calcular as métricas
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)

    # Verifica se o modelo suporta predict_proba para calcular auc-roc
    try:
        y_proba = modelo.predict_proba(X)[:,1]
        auc = roc_auc_score(y, y_proba)
    except AttributeError:
        auc = np.nan  # Se o modelo não tiver predict_proba

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'auc': auc
    }

"""
Gera o DataFrame de submissão para o Kaggle a partir das previsões do modelo

Parâmetros:
- modelo: Modelo treinado
- X_test: Conjunto de features do teste
- df_submission: DataFrame contendo as colunas 'Id' e a estrutura para 'Predicted'

Retorna:
- df_kaggle (pd.DataFrame): DataFrame com os IDs e as probabilidades preditas para a classe "Anormal"
"""
def gerar_metrica_kaggle(modelo, X_test, df_submission):
    # Obter as probabilidades da classe "Anormal"
    y_pred_prob = modelo.predict_proba(X_test)[:,1]

    df_kaggle = df_submission.copy()
    df_kaggle['Predicted'] = y_pred_prob
    return df_kaggle


"""
Realiza validação cruzada no modelo e retorna a média das pontuações obtidas

Parâmetros:
- modelo: Modelo a ser avaliado 
- X: Conjunto de features de entrada
- y: Labels reais
- cv (int): Número de folds para a validação cruzada 
- scoring (str): Métrica utilizada para avaliação 

Retorna:
- float: Média das pontuações obtidas na validação cruzada
"""
def validar_modelo_cv(modelo, X, y, cv=5, scoring='roc_auc'):
    scores = cross_val_score(modelo, X, y, cv=cv, scoring=scoring)
    return scores.mean()

"""
Carrega um modelo treinado a partir de um arquivo

Parâmetros:
- nome_arquivo (str): Caminho e nome do arquivo que contém o modelo salvo

Retorna:
- modelo: Modelo carregado
"""
def carregar_modelo(nome_arquivo):
    modelo = joblib.load(nome_arquivo)
    print(f"Modelo carregado de {nome_arquivo}")
    return modelo

"""
Plota a curva ROC para cada modelo, mostrando o desempenho em termos de AUC

Parâmetros:
- modelos (list): Lista de modelos treinados
- nomes_modelos (list): Lista com os nomes dos modelos
- X_test (np.array ou DataFrame): Conjunto de features do teste
- y_test (np.array): Labels reais correspondentes ao conjunto de teste
"""
def plot_roc_curves(modelos, nomes_modelos, X_test, y_test):
    plt.figure(figsize=(10, 8))
    
    # Itera sobre os modelos e seus respectivos nomes para calcular e plotar a curva ROC
    for modelo, nome in zip(modelos, nomes_modelos):
        try:
            # Obtém as probabilidades da classe positiva
            y_proba = modelo.predict_proba(X_test)[:, 1]
            # Calcula as taxas de Falso Positivo e Verdadeiro Positivo 
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            # Calcula a área sob a curva 
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{nome} (AUC = {roc_auc:.4f})')
        except Exception as e:
            print(f"Erro ao gerar a curva ROC para {nome}: {e}")
    
    # Linha diagonal representando a performance de um classificador aleatório
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title('Curva ROC dos Modelos')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

"""
Plota a curva de aprendizado de um modelo

Parâmetros:
- modelo: Modelo a ser avaliado
- X (np.array ou DataFrame): Conjunto de features
- y (np.array): Labels reais
- cv (int): Número de folds para validação cruzada (padrão = 5)
- scoring (str): Métrica para avaliação (padrão = 'accuracy')
- train_sizes (array): Vetor com os tamanhos de treino a serem avaliados (padrão = np.linspace(0.1, 1.0, 10))

"""
def plot_learning_curve(modelo, X, y, cv=5, scoring='roc_auc', train_sizes=np.linspace(0.2, 1.0, 5)):
    train_sizes, train_scores, val_scores = learning_curve(modelo, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    val_scores_mean   = np.mean(val_scores, axis=1)
    val_scores_std    = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(12, 8))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Treino', markersize=8)
    plt.plot(train_sizes, val_scores_mean, 'o-', color='green', label='Validação', markersize=8)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.15, color='blue')
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.15, color='green')
    
    plt.xlabel('Número de Amostras de Treino')
    plt.ylabel('AUC-ROC')
    plt.title('Curva de Aprendizado')
    plt.ylim(0.90, 1.00)  
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.show()



"""
Plota a matriz de confusão para um modelo

Parâmetros:
- modelo: Modelo treinado
- X_test (np.array ou DataFrame): Conjunto de features do teste
- y_test (np.array): Labels reais do teste
- normalize (bool): Se True, a matriz será normalizada (padrão = False)

"""
def plot_confusion_matrix(modelo, X_test, y_test, normalize=False):
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize='true' if normalize else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='.2f' if normalize else 'd')
    plt.title('Matriz de Confusão' + (' (Normalizada)' if normalize else ''))
    plt.show()


"""
Plota um boxplot dos scores obtidos na validação cruzada

Parâmetros:
- cv_scores (array-like): Array ou lista com os scores de cada fold da validação cruzada
- nome_modelo (str): Nome do modelo para identificação no gráfico

"""
def plot_cv_boxplot(cv_scores, nome_modelo):
    plt.figure(figsize=(6, 8))
    plt.boxplot(cv_scores, patch_artist=True, labels=[nome_modelo])
    plt.ylabel('Score')
    plt.title('Dispersão dos Scores - Validação Cruzada')
    plt.grid(alpha=0.3)
    plt.show()