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

# Arquivo com todas as funções e códigos referentes ao pré-processamento
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

"""
Função para unificar as etapas do pré-processamento dos dados

Etapas:
1. Remoção de colunas irrelevantes
2. Conversão de tipos 
3. Tratamento de inconsistências 
4. Imputação de valores faltantes
5. Criação de features
6. Codificação de variáveis categóricas
7. Tratamento de outliers
8. Escalonamento das features numéricas

Parâmetros:
- df (pd.DataFrame): DataFrame a ser pré-processado

Retorna:
- df (pd.DataFrame): DataFrame pré-processado
- scaler (RobustScaler): Objeto RobustScaler ajustado às colunas numéricas
"""
def pipeline_pre_processamento(df, fit_scaler=True, scaler=None):

    print("=== Iniciando Pipeline de Pre-Processamento ===")

    # 1. Remoção de colunas irrelevantes
    # Definir colunas a remover de acordo com a análise 
    colunas_remover = ['Id', 'Convenio', 'Atendimento', 'DN', 'PPA']

    for col in colunas_remover:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # 2. Conversão de tipos
    df = converter_tipos(df)

    # 3. Tratamento de inconsistências
    df = tratar_inconsistencias(df)

    # 4. Imputação de valores faltantes
    df = imputar_valores_faltantes(df)

    # 5. Criação de features
    df = criar_features_adicionais(df)

    # 6. Codificação de variáveis categóricas
    df = codificar_categorias(df)

    # 7. Tratamento de outliers
    colunas_outliers = ["Peso", "Altura", "IMC", "IDADE", "PA SISTOLICA", "PA DIASTOLICA"]
    if 'FC' in df.columns and np.issubdtype(df['FC'].dtype, np.number):
        colunas_outliers.append('FC')  
    df = tratar_outliers(df, colunas=colunas_outliers)

    # 8. Escalonamento de features numéricas
    colunas_para_escalar = ["Peso", "Altura", "IMC", "IDADE", "PA SISTOLICA", "PA DIASTOLICA"]
    if 'FC' in df.columns and np.issubdtype(df['FC'].dtype, np.number):
        colunas_para_escalar.append('FC')
    if 'FC_por_Idade' in df.columns:
        colunas_para_escalar.append('FC_por_Idade')

    if scaler is None:
        scaler = RobustScaler()

    if fit_scaler:
        print("[INFO] Ajustando (fit) o scaler nas colunas numéricas.")
        df[colunas_para_escalar] = scaler.fit_transform(df[colunas_para_escalar])
    else:
        print("[INFO] Aplicando (transform) o scaler já treinado.")
        df[colunas_para_escalar] = scaler.transform(df[colunas_para_escalar])


    print("=== Pipeline de Pre-Processamento Concluído ===")
    return df, scaler

"""
Converte colunas numéricas para tipo float

Parâmetros:
- df (pd.DataFrame): DataFrame a ser convertido

Retorna:
- df (pd.DataFrame): DataFrame com tipos de dados convertidos
"""
def converter_tipos(df):
    print("[INFO] (converter_tipos) Convertendo colunas numéricas...")

    colunas_numericas = ["Peso", "Altura", "IMC", "IDADE", "PA SISTOLICA", "PA DIASTOLICA", "FC"]
    for col in colunas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"[INFO] Coluna '{col}' convertida para numérico")

    return df

"""
Ajusta valores inconsistentes ou fisicamente impossíveis, substituindo por medianas

Parâmetros:
- df (pd.DataFrame): DataFrame a ser tratado

Retorna:
- df (pd.DataFrame): DataFrame com inconsistências tratadas
"""
def tratar_inconsistencias(df):
    print("[INFO] (tratar_inconsistencias) Ajustando inconsistências...")

    # Idade: 0 ano <= idade <= 20 anos
    if 'IDADE' in df.columns:
        cond_neg_idade = df['IDADE'] < 0
        cond_alt_idade = df['IDADE'] > 20
        if cond_neg_idade.sum() > 0:
            med_idade = df.loc[~cond_neg_idade, 'IDADE'].median()
            df.loc[cond_neg_idade, 'IDADE'] = med_idade
            print(f"[INFO] Corrigidos {cond_neg_idade.sum()} valores negativos de 'IDADE' via mediana {med_idade}")
        if cond_alt_idade.sum() > 0:
            med_idade = df.loc[~cond_alt_idade, 'IDADE'].median()
            df.loc[cond_alt_idade, 'IDADE'] = med_idade
            print(f"[INFO] Corrigidos {cond_alt_idade.sum()} valores de 'IDADE' > 20 via mediana {med_idade}")

    # Altura: 25 cm <= altura <= 200 cm
    if 'Altura' in df.columns:
        cond_altura_low = df['Altura'] < 25
        cond_altura_high = df['Altura'] > 200
        if cond_altura_low.sum() > 0:
            med_altura = df.loc[~cond_altura_low, 'Altura'].median()
            df.loc[cond_altura_low, 'Altura'] = med_altura
            print(f"[INFO] Corrigidos {cond_altura_low.sum()} valores de 'Altura' < 25cm via mediana {med_altura}")
        if cond_altura_high.sum() > 0:
            med_altura = df.loc[~cond_altura_high, 'Altura'].median()
            df.loc[cond_altura_high, 'Altura'] = med_altura
            print(f"[INFO] Corrigidos {cond_altura_high.sum()} valores de 'Altura' > 200cm via mediana {med_altura}")

    # Peso: 2 kg <= peso <= 150 kg 
    if 'Peso' in df.columns:
        cond_peso_low = df['Peso'] < 2
        cond_peso_high = df['Peso'] > 150
        if cond_peso_low.sum() > 0:
            med_peso = df.loc[~cond_peso_low, 'Peso'].median()
            df.loc[cond_peso_low, 'Peso'] = med_peso
            print(f"[INFO] Corrigidos {cond_peso_low.sum()} valores de 'Peso' < 2kg via mediana {med_peso}")
        if cond_peso_high.sum() > 0:
            med_peso = df.loc[~cond_peso_high, 'Peso'].median()
            df.loc[cond_peso_high, 'Peso'] = med_peso
            print(f"[INFO] Corrigidos {cond_peso_high.sum()} valores de 'Peso' > 100kg via mediana {med_peso}")

    # PA SISTOLICA: 70 <= PA SISTOLICA <= 200
    if 'PA SISTOLICA' in df.columns:
        cond_pa_low = df['PA SISTOLICA'] < 70
        cond_pa_high = df['PA SISTOLICA'] > 200
        if cond_pa_low.sum() > 0:
            med_pa_sist = df.loc[~cond_pa_low, 'PA SISTOLICA'].median()
            df.loc[cond_pa_low, 'PA SISTOLICA'] = med_pa_sist
            print(f"[INFO] Corrigidos {cond_pa_low.sum()} valores de 'PA SISTOLICA' < 70 via mediana {med_pa_sist}")
        if cond_pa_high.sum() > 0:
            med_pa_sist = df.loc[~cond_pa_high, 'PA SISTOLICA'].median()
            df.loc[cond_pa_high, 'PA SISTOLICA'] = med_pa_sist
            print(f"[INFO] Corrigidos {cond_pa_high.sum()} valores de 'PA SISTOLICA' > 200 via mediana {med_pa_sist}")

    # PA DIASTOLICA: 40 <= PA DIASTOLICA <= 120
    if 'PA DIASTOLICA' in df.columns:
        cond_pa_dia_low = df['PA DIASTOLICA'] < 40
        cond_pa_dia_high = df['PA DIASTOLICA'] > 120
        if cond_pa_dia_low.sum() > 0:
            med_pa_dia = df.loc[~cond_pa_dia_low, 'PA DIASTOLICA'].median()
            df.loc[cond_pa_dia_low, 'PA DIASTOLICA'] = med_pa_dia
            print(f"[INFO] Corrigidos {cond_pa_dia_low.sum()} valores de 'PA DIASTOLICA' < 40 via mediana {med_pa_dia}")
        if cond_pa_dia_high.sum() > 0:
            med_pa_dia = df.loc[~cond_pa_dia_high, 'PA DIASTOLICA'].median()
            df.loc[cond_pa_dia_high, 'PA DIASTOLICA'] = med_pa_dia
            print(f"[INFO] Corrigidos {cond_pa_dia_high.sum()} valores de 'PA DIASTOLICA' > 120 via mediana {med_pa_dia}")

    # FC: 30 <= FC <= 200
    if 'FC' in df.columns:
        cond_fc_low = df['FC'] < 30
        cond_fc_high = df['FC'] > 200
        if cond_fc_low.sum() > 0:
            med_fc = df.loc[~cond_fc_low, 'FC'].median()
            df.loc[cond_fc_low, 'FC'] = med_fc
            print(f"[INFO] Corrigidos {cond_fc_low.sum()} valores de 'FC' < 30 via mediana {med_fc}")
        if cond_fc_high.sum() > 0:
            med_fc = df.loc[~cond_fc_high, 'FC'].median()
            df.loc[cond_fc_high, 'FC'] = med_fc
            print(f"[INFO] Corrigidos {cond_fc_high.sum()} valores de 'FC' > 200 via mediana {med_fc}")

    return df

"""
Imputa valores faltantes nas colunas numéricas com a mediana e nas colunas categóricas com a moda

Parâmetros:
- df (pd.DataFrame): DataFrame a ser tratado

Retorna:
- df (pd.DataFrame): DataFrame com valores faltantes imputados
"""
def imputar_valores_faltantes(df):
    print("[INFO] (imputar_valores_faltantes) Imputando NaNs...")

    # Separar colunas numéricas
    colunas_num = ["Peso", "Altura", "IMC", "IDADE", "PA SISTOLICA", "PA DIASTOLICA"]
    if 'FC' in df.columns and np.issubdtype(df['FC'].dtype, np.number):
        colunas_num.append('FC')

    # Separar colunas categóricas 
    colunas_cat = []
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['CLASSE']:  # CLASSE não será imputada
            colunas_cat.append(col)

    if 'Peso' in df.columns and 'Altura' in df.columns and 'IMC' in df.columns:
        print("[INFO] Recalculando 'IMC' onde possível...")
        cond_imc = df['IMC'].isna() & df['Peso'].notna() & (df['Altura'] > 0)
        df.loc[cond_imc, 'IMC'] = df.loc[cond_imc, 'Peso'] / ((df.loc[cond_imc, 'Altura'] / 100)**2)

    # Imputar colunas numéricas com mediana
    imp_mediana = SimpleImputer(strategy='median')
    df[colunas_num] = imp_mediana.fit_transform(df[colunas_num])
    print(f"[INFO] Colunas numéricas {colunas_num} imputadas com mediana.")

    # Imputar colunas categóricas com moda
    if len(colunas_cat) > 0:
        imp_moda = SimpleImputer(strategy='most_frequent')
        df[colunas_cat] = imp_moda.fit_transform(df[colunas_cat])
        print(f"[INFO] Colunas categóricas {colunas_cat} imputadas com moda.")

    return df

"""
Cria features adicionais relevantes para a análise de cardiopatias em crianças

Features criadas:
- Faixa_Etaria: Segmentação da idade em faixas etárias
- Faixa_IMC: Classificação do IMC em categorias
- FC_por_Idade: Frequência cardíaca normalizada pela idade

Parâmetros:
- df (pd.DataFrame): DataFrame original

Retorna:
- df (pd.DataFrame): DataFrame com novas features adicionadas
"""
def criar_features_adicionais(df):
    print("[INFO] (criar_features_adicionais) Criando features adicionais...")

    # Faixa etária
    if 'IDADE' in df.columns:
        df['Faixa_Etaria'] = pd.cut(df['IDADE'], bins=[0, 2, 6, 12, 20], labels=['0-2', '2-6', '6-12', '12-20'], right=False)
        print("[INFO] Feature 'Faixa_Etaria' criada.")
    
    # Faixa de IMC
    if 'IMC' in df.columns:
        df['Faixa_IMC'] = pd.cut(df['IMC'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['abaixo_peso', 'peso_normal', 'sobrepeso', 'obesidade'], right=False)
        print("[INFO] Feature 'Faixa_IMC' criada.")

    # Frequência Cardíaca normalizada pela idade
    if 'FC' in df.columns and 'IDADE' in df.columns:
        df['FC_por_Idade'] = df['FC'] / (df['IDADE'] + 1)  # +1 para evitar divisão por zero
        print("[INFO] Feature 'FC_por_Idade' criada.")

    return df

"""
Aplica codificações (One-Hot Encoding) para colunas categóricas relevantes

Parâmetros:
- df (pd.DataFrame): DataFrame a ser codificado

Retorna:
- df (pd.DataFrame): DataFrame com variáveis categóricas codificadas
"""
def codificar_categorias(df):
    print("[INFO] (codificar_categorias) Aplicando codificações...")

    # Mapas de conversão para padronizar categorias e evitar colunas inconsistentes
    map_sexo = {
        'm': 'masculino',
        'M': 'masculino',
        'masculino': 'masculino',
        'male': 'masculino',
        'f': 'feminino',
        'F': 'feminino',
        'female': 'feminino',
        'feminino': 'feminino',
        'Feminino': 'feminino',
        'indeterminado': 'indeterminado',
        'Indeterminado': 'indeterminado'
    }
    
    df['SEXO'] = df['SEXO'].map(map_sexo)

    map_sopro = {
        'Contínuo': 'continuo',
        'contínuo': 'continuo',
        'Sistolico e diastólico': 'sistolico_e_diastolico',
        'Sistólico': 'sistolico',
        'sistólico': 'sistolico',
        'diastólico': 'diastolico',
        'Diastólico': 'diastolico',
        'ausente': 'ausente',
        'Ausente': 'ausente'
    }

    df['SOPRO'] = df['SOPRO'].map(map_sopro)

    map_b2 = {
        'Desdob fixo': 'desdob_fixo',
        'Hiperfonética': 'hiperfonetica',
        'Normal': 'normal',
        'Outro': 'outro',
        'Única': 'unica'
    }

    df['B2'] = df['B2'].map(map_b2)

    map_pulsos = {
        'AMPLOS': 'amplos',
        'Amplos': 'amplos',
        'Diminuídos ': 'diminuidos',
        'Femorais diminuidos': 'femorais_diminuidos',
        'NORMAIS': 'normais',
        'Normais': 'normais',
        'Outro': 'outro'
    }

    df['PULSOS'] = df['PULSOS'].map(map_pulsos)

    # Colunas que serão codificadas com One-Hot
    colunas_one_hot = ['SOPRO', 'B2', 'PULSOS', 'SEXO', 'Faixa_Etaria', 'Faixa_IMC', 'MOTIVO1', 'MOTIVO2', 'HDA 1', 'HDA2']

    # One-Hot Encoding
    to_encode = [c for c in colunas_one_hot if c in df.columns]
    if to_encode:
        print(f"[INFO] Aplicando One-Hot Encoding em {to_encode}")
        df = pd.get_dummies(df, columns=to_encode)
        print(f"[INFO] One-Hot Encoding aplicado em {to_encode}")

    return df

"""
Trata outliers nas colunas especificadas, trocando pela mediana e criando indicadores de outliers

Parâmetros:
- df (pd.DataFrame): DataFrame a ser tratado
- colunas (list): Lista de colunas numéricas para tratar outliers
- lower_percentile (int): Percentil inferior para capagem
- upper_percentile (int): Percentil superior para capagem

Retorna:
- df (pd.DataFrame): DataFrame com outliers tratados e indicadores adicionados
"""
def tratar_outliers(df, colunas, lower_percentile=1, upper_percentile=99):

    print("[INFO] (tratar_outliers_substituir_mediana) Substituindo outliers pela mediana e criando colunas 'is_outlier_<col>'...")

    for col in colunas:
        if col not in df.columns:
            continue

        low_val = df[col].quantile(lower_percentile / 100)
        high_val = df[col].quantile(upper_percentile / 100)

        med = df[col].median()

        # Criar coluna indicadora
        is_outlier_col = f'is_outlier_{col}'
        df[is_outlier_col] = ((df[col] < low_val) | (df[col] > high_val)).astype(int)

        # Substituir outliers pela mediana
        df.loc[df[col] < low_val, col] = med
        df.loc[df[col] > high_val, col] = med

        num_outliers = df[is_outlier_col].sum()
        print(f"  -> Coluna '{col}': {num_outliers} outliers substituídos pela mediana {med}.")

    return df

"""
Aplica RobustScaler nas colunas numéricas especificadas

Parâmetros:
- df (pd.DataFrame): DataFrame a ser tratado
- colunas_para_escalar (list): Lista de colunas numéricas para escalonamento

Retorna:
- df (pd.DataFrame): DataFrame com colunas escalonadas
- scaler (RobustScaler): Objeto RobustScaler ajustado
"""
def escalar_features(df, colunas_para_escalar):

    print("[INFO] (escalar_features) Escalonando colunas numéricas...")

    scaler = RobustScaler()
    df[colunas_para_escalar] = scaler.fit_transform(df[colunas_para_escalar])
    print(f"[INFO] Escalonamento aplicado em {colunas_para_escalar}")

    return df, scaler

"""
Exporta o DataFrame pré-processado para um arquivo CSV

Parâmetros:
- df (pd.DataFrame): DataFrame a ser exportado
- caminho (str): Caminho do arquivo de saída
"""
def exportar_dados(df, caminho='pre_processado.csv'):

    df.to_csv(caminho, index=False)
    print(f"[INFO] Dados pré-processados exportados em '{caminho}'")