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

# Arquivo com todas as funções e códigos referentes à análise exploratória

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import os
import missingno as msno

"""
Exibe as primeiras linhas, informações gerais e estatísticas descritivas do Dataframe

Parâmetros:
    - df (pd.DataFrame): DataFrame a ser analisado
"""
def visao_geral(df):

    # Definição do números de linhas a ser exibido
    max_linhas = 10

    # Exibição das primeiras linhas do Dataframe
    print("=== Primeiras Linhas do DataFrame ===")
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=df.columns,                      
            fill_color='#6A78A8',                   
            font=dict(color='white', size=12),      
            align='center',                         
            line_color='black'                      
        ),
        cells=dict(
            values=[df[col][:max_linhas] for col in df.columns],  
            fill_color='#ffffff',                                 
            font=dict(color='black', size=11),                    
            align='center',                                       
            line_color='black'                                    
        )
    )])

    fig.update_layout(
        width=1920,  
        height=540, 
        margin=dict(l=20, r=20, t=20, b=20)  
    )

    fig.show()
    
    # Exibição das dimensões
    print("\n=== Dimensões do DataFrame ===")
    print(f"Linhas: {df.shape[0]}, Colunas: {df.shape[1]}")
    
    # Exibição das informações gerais
    print("\n=== Informações do DataFrame ===")
    print(df.info())
    
    # Exibição das estatísticas descritivas
    print("\n=== Estatísticas Descritivas (Numéricas) ===")
    estatisticas = df.describe(include=[np.number])
    print(estatisticas)
    
    # Salvar estatísticas em arquivo
    estatisticas.to_csv('reports/estatisticas_descritivas.csv')
    print("\nEstatísticas descritivas salvas em 'estatisticas_descritivas.csv'")
    
    # Descrição das colunas
    descricao_colunas = {
        "Id": "Identificador único do paciente",
        "Peso": "Peso do paciente em kg",
        "Altura": "Altura do paciente em cm",
        "IMC": "Índice de Massa Corporal (kg/m²)",
        "Atendimento": "Data do atendimento",
        "DN": "Data de nascimento",
        "IDADE": "Idade do paciente em anos (pode ser fracionária)",
        "Convenio": "Convênio do paciente",
        "PULSOS": "Pulsos do paciente",
        "PA SISTOLICA": "Pressão Arterial Sistólica",
        "PA DIASTOLICA": "Pressão Arterial Diastólica",
        "PPA": "Pressão de Pulso Arterial",
        "B2": "Possível variável clínica adicional",
        "SOPRO": "Presença de sopro cardíaco",
        "FC": "Frequência Cardíaca",
        "HDA 1": "História da Doença Atual 1",
        "HDA2": "História da Doença Atual 2",
        "SEXO": "Sexo do paciente (M/F)",
        "MOTIVO1": "Motivo principal da consulta",
        "MOTIVO2": "Motivo secundário da consulta"
    }
    
    print("\n=== Descrição das Colunas ===")
    for coluna, descricao in descricao_colunas.items():
        print(f"{coluna}: {descricao}")
    
    print("\n")

"""
Verifica inconsistências no DataFrame gerando relatórios e plots

Parâmetros:
    - df (pd.DataFrame): DataFrame a ser verificado
    - salvar_relatorio (bool): Se True, salva um relatório CSV das inconsistências
    - caminho_relatorio (str): Caminho para salvar o relatório CSV
    - salvar_plots (bool): Se True, salva gráficos das inconsistências
    - caminho_plots (str): Diretório para salvar os gráficos

Retorna:
- inconsistencias (dict): Dicionário das inconsistências encontradas
"""
def verificar_integridade(df, salvar_relatorio=True, caminho_relatorio='reports/inconsistencias_report.csv', salvar_plots=True, caminho_plots='plots/'):
    inconsistencias = {}

    # Verificações de integridade
    
    # Idade negativa
    neg_idade = df[df['IDADE'] < 0]
    if not neg_idade.empty:
        inconsistencias['IDADE_negativa'] = neg_idade
    
    # Peso ou Altura <= 0
    zero_peso = df[df['Peso'] <= 0]
    zero_altura = df[df['Altura'] <= 0]
    if not zero_peso.empty:
        inconsistencias['Peso_zero'] = zero_peso
    if not zero_altura.empty:
        inconsistencias['Altura_zero'] = zero_altura
    
    # PA Sistólica < PA Diastólica 
    cond_pa = (
        df['PA SISTOLICA'].notna() & 
        df['PA DIASTOLICA'].notna() &
        (df['PA SISTOLICA'] < df['PA DIASTOLICA'])
    )
    pa_inconsistentes = df[cond_pa]
    if not pa_inconsistentes.empty:
        inconsistencias['PA_incoerente'] = pa_inconsistentes
    
    if inconsistencias:
        # Criação diretório para salvar plots caso não exista
        if salvar_plots and not os.path.exists(caminho_plots):
            os.makedirs(caminho_plots)
        
        # Resumo das inconsistências
        resumo = {tipo: len(dados) for tipo, dados in inconsistencias.items()}
        resumo_df = pd.DataFrame(list(resumo.items()), columns=['Tipo de Inconsistência', 'Quantidade'])
        
        print("\n=== Resumo das Inconsistências Encontradas ===")
        print(resumo_df)
        print("\n")
        
        # Salvar relatório em CSV
        if salvar_relatorio:
            resumo_df.to_csv(caminho_relatorio, index=False)
            print(f"\nRelatório de inconsistências salvo em '{caminho_relatorio}'")
        
        # Plotar gráfico de barras das inconsistências
        if salvar_plots:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Quantidade', y='Tipo de Inconsistência', data=resumo_df, palette='viridis')
            plt.title('Tipos de Inconsistências Encontradas')
            plt.xlabel('Quantidade')
            plt.ylabel('Tipo de Inconsistência')
            plt.tight_layout()
            caminho_plot = os.path.join(caminho_plots, 'inconsistencias_barras.png')
            plt.savefig(caminho_plot)
            plt.close()
            print(f"\nGráfico de inconsistências salvo em '{caminho_plot}'")
        
        # Salvar detalhes de cada inconsistência
        for tipo, dados in inconsistencias.items():
            if salvar_relatorio:
                arquivo_detalhe = os.path.splitext(caminho_relatorio)[0] + f'_{tipo}.csv'
                dados.to_csv(arquivo_detalhe, index=False)
                print(f"\nDetalhes da inconsistência '{tipo}' salvos em '{arquivo_detalhe}'")
        
    else:
        print("\nNenhuma inconsistência clínica encontrada.")
    
    return inconsistencias


"""
Plota e salva um mapa de valores ausentes no DataFrame
Salva o gráfico em 'plots/missing_values.png'

Parâmetros:
    - df (pd.DataFrame): DataFrame a ser analisado

"""
def plot_missing_values(df):
    plt.figure(figsize=(12, 6))
    msno.matrix(df)
    plt.title('Mapa de Valores Ausentes')
    plt.savefig('plots/missing_values.png')
    plt.close()
    print("Mapa de valores ausentes salvo em 'plots/missing_values.png'")

"""
Plota e salva um heatmap de valores ausentes no DataFrame
Salva o gráfico em 'plots/missing_values_heatmap.png'

Parâmetros:
    - df (pd.DataFrame): DataFrame a ser analisado
"""
def plot_missing_heatmap(df):
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Heatmap de Valores Ausentes')
    plt.savefig('plots/missing_values_heatmap.png')
    plt.close()
    print("Heatmap de valores ausentes salvo em 'plots/missing_values_heatmap.png'")

"""
Plota e salva boxplots para as colunas numéricas passadas
Salva cada plot em 'plots/boxplot_<coluna>.png'

Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados
    - colunas (list): Lista de colunas numéricas para plotar
"""
def plot_boxplots(df, colunas):
    for col in colunas:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot de {col}')
        plt.savefig(f'plots/boxplot_{col}.png')
        plt.close()
        print(f"Boxplot de {col} salvo em 'plots/boxplot_{col}.png'")

"""
Calcula e exibe skewness (assimetria), kurtosis (curtose) e matriz de correlação para as colunas numéricas
Salva a matriz de correlação em 'plots/matriz_correlacao.png'

Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados
    - colunas_numericas (list): Lista de colunas numéricas para análise
"""
def analise_estatistica(df, colunas_numericas):

    print("\n=== Skewness (Assimetria) ===")
    skew = df[colunas_numericas].skew()
    print(skew)
    
    print("\n=== Kurtosis (Curtose) ===")
    kurt = df[colunas_numericas].kurtosis()
    print(kurt)
    
    # Matriz de Correlação
    corr = df[colunas_numericas].corr()
    print("\n=== Matriz de Correlação ===")
    print(corr)
    
    # Plotar heatmap da correlação
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlação entre Variáveis Numéricas')
    plt.savefig('plots/matriz_correlacao.png')
    plt.close()
    print("\nMatriz de correlação salva em 'plots/matriz_correlacao.png'")

"""
Plota e salva histogramas com KDE para as colunas numéricas especificadas
Salva cada plot em 'plots/distribuicao_<coluna>.png'

Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados
    - colunas_numericas (list): Lista de colunas numéricas para plotar
"""
def plot_distribuicoes(df, colunas_numericas):
    for col in colunas_numericas:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribuição de {col}')
        plt.savefig(f'plots/distribuicao_{col}.png')
        plt.close()
        print(f"\nDistribuição de {col} salva em 'plots/distribuicao_{col}.png'")

"""
Plota e salva violin plots para as colunas numéricas segmentadas por uma coluna categórica
Salva cada plot em 'plots/violin_<coluna>.png'

Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados
    - colunas_numericas (list): Lista de colunas numéricas para plotar
    - coluna_categorica (str): Coluna categórica para segmentação
"""
def plot_violin(df, colunas_numericas, coluna_categorica):
    for col in colunas_numericas:
        plt.figure(figsize=(8, 6))
        sns.violinplot(x=coluna_categorica, y=col, data=df, palette='Set2')
        plt.title(f'Violin Plot de {col} por {coluna_categorica}')
        plt.savefig(f'plots/violin_{col}.png')
        plt.close()
        print(f"\nViolin plot de {col} por {coluna_categorica} salvo em 'plots/violin_{col}.png'")

"""
Plota e salva um pairplot das colunas numéricas, colorido por uma coluna selecionada
Salva o plot em 'plots/pairplot_<coluna_selecionada>.png'

Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados
    - colunas_numericas (list): Lista de colunas numéricas para incluir no pairplot
    - coluna_selecionada (str): Coluna para colorir os pontos no pairplot
"""
def plot_pairplot(df, colunas_numericas, coluna_selecionada):
    sns.pairplot(df, vars=colunas_numericas, hue=coluna_selecionada, palette='viridis')
    plt.suptitle(f'Pairplot das Variáveis Numéricas por {coluna_selecionada}', y=1.02)
    plt.savefig(f'plots/pairplot_{coluna_selecionada}.png')
    plt.close()
    print(f"\nPairplot salvo em 'plots/pairplot_{coluna_selecionada}.png'")

"""
Segmenta o DataFrame por faixas etárias e plota boxplots das variáveis numéricas para cada faixa
Salva os plots em 'plots/faixa_etaria_<coluna>.png'

Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados
    - colunas_numericas (list): Lista de colunas numéricas para plotar
"""
def segmentar_por_faixa_etaria(df, colunas_numericas):
    # Definição de faixas etárias
    bins = [0, 2, 6, 12, 19]
    labels = ['0-2 anos', '3-6 anos', '7-12 anos', '13-19 anos']
    df['FaixaEtaria'] = pd.cut(df['IDADE'], bins=bins, labels=labels, right=False)
    
    # Plotar boxplots por faixa etária
    for col in colunas_numericas:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='FaixaEtaria', y=col, data=df, palette='Set3')
        plt.title(f'Boxplot de {col} por Faixa Etária')
        plt.savefig(f'plots/faixa_etaria_{col}.png')
        plt.close()
        print(f"\nBoxplot de {col} por Faixa Etária salvo em 'plots/faixa_etaria_{col}.png'")

"""
Compara as distribuições das variáveis numéricas entre pacientes com e sem sopro
Salva os plots em 'plots/comparacao_sopro_<coluna>.png'

Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados
    - colunas_numericas (list): Lista de colunas numéricas para comparar
"""
def comparar_sopro(df, colunas_numericas):
    for col in colunas_numericas:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='SOPRO', y=col, data=df, palette='Set2')
        plt.title(f'Comparação de {col} por Presença de Sopro Cardíaco')
        plt.savefig(f'plots/comparacao_sopro_{col}.png')
        plt.close()
        print(f"\nComparação de {col} por Sopro Cardíaco salva em 'plots/comparacao_sopro_{col}.png'")

"""
Plota um grid de histogramas (com KDE) para todas as colunas numéricas especificadas

Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados
    colunas_numericas (list): Lista de colunas numéricas a serem plotadas
"""
def histograma_geral(df, colunas_numericas):
    
    n_cols = 3  
    n_rows = int(np.ceil(len(colunas_numericas) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(colunas_numericas):
        sns.histplot(df[col], kde=True, bins=30, color='skyblue', ax=axes[i])
        axes[i].set_title(f'Distribuição de {col}')
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

"""
Exibe um heatmap da matriz de correlação para as colunas numéricas especificadas

Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados
    colunas_numericas (list): Lista de colunas numéricas para análise
"""
def exibir_heatmap_correlacao(df, colunas_numericas):
    
    corr = df[colunas_numericas].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlação entre Variáveis Numéricas')
    plt.tight_layout()
    plt.show()

"""
Plota boxplots de todas as colunas numéricas em uma única figura, sem exibir os outliers

Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados
    colunas_numericas (list): Lista de colunas numéricas a serem plotadas
"""
def boxplots_unicos(df, colunas_numericas):

    # Converte cada coluna para numérico e remove valores nulos
    dados = [pd.to_numeric(df[col], errors='coerce').dropna().values for col in colunas_numericas]
    plt.figure(figsize=(12, 6))
    plt.boxplot(dados, showfliers=False, labels=colunas_numericas)
    plt.title('Boxplots das Variáveis Numéricas')
    plt.ylabel('Valores')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

"""
Plota um gráfico de barras mostrando a distribuição dos valores de uma variável categórica

Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados
    coluna (str): Nome da coluna categórica
"""
def plot_valores_categoricos(df, coluna):
    
    plt.figure(figsize=(8, 6))
    contagem = df[coluna].value_counts()
    sns.barplot(x=contagem.index, y=contagem.values, palette='pastel')
    plt.title(f'Distribuição da variável {coluna}')
    plt.xlabel(coluna)
    plt.ylabel('Contagem')
    plt.tight_layout()
    plt.show()

"""
Exibe um plot das inconsistências encontradas no DataFrame

Parâmetros:
    inconsistencias (dict): Dicionário com os tipos de inconsistências e seus respectivos DataFrames
"""
def exibir_inconsistencias(inconsistencias):
    
    if inconsistencias:
        resumo = {tipo: len(dados) for tipo, dados in inconsistencias.items()}
        resumo_df = pd.DataFrame(list(resumo.items()), columns=['Tipo de Inconsistência', 'Quantidade'])
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Quantidade', y='Tipo de Inconsistência', data=resumo_df, palette='viridis')
        plt.title('Resumo das Inconsistências Encontradas')
        plt.xlabel('Quantidade')
        plt.ylabel('Tipo de Inconsistência')
        plt.tight_layout()
        plt.show()
    else:
        print("Nenhuma inconsistência clínica encontrada.")
        
