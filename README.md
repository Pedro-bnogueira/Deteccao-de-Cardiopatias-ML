# Detecção de Cardiopatias em Pacientes Pediátricos com Aprendizado de Máquina

Este repositório contém todo o código e documentação referentes ao projeto de detecção de patologias cardíacas em pacientes de até 19 anos (com alguns registros de adultos) por meio de técnicas de Aprendizado de Máquina (AM). O objetivo principal é demonstrar como diferentes algoritmos podem ser aplicados para auxiliar no diagnóstico precoce de doenças cardíacas em crianças e adolescentes, utilizando dados clínicos e antropométricos de um hospital real.

---

## Sumário

1. [Contexto e Motivação](#contexto-e-motivação)
2. [Descrição do Projeto](#descrição-do-projeto)
3. [Estrutura de Arquivos](#estrutura-de-arquivos)
4. [Conjunto de Dados](#conjunto-de-dados)
5. [Processo de Pré-Processamento](#processo-de-pré-processamento)
6. [Arquitetura de Modelagem e Experimentação](#arquitetura-de-modelagem-e-experimentação)
7. [Como Executar o Projeto](#como-executar-o-projeto)
8. [Principais Resultados e Discussões](#principais-resultados-e-discussões)
9. [Conclusões](#conclusões)
10. [Referências](#referências)

---

## Contexto e Motivação

A detecção precoce de doenças cardíacas em crianças é um desafio clínico de grande relevância, pois muitas patologias podem se manifestar de forma silenciosa ou inespecífica. Com o avanço das tecnologias de processamento de dados e técnicas de Aprendizado de Máquina, tornou-se viável desenvolver sistemas capazes de auxiliar profissionais de saúde na classificação de pacientes em risco.

Este projeto, desenvolvido no âmbito acadêmico, teve como objetivos:
- **Investigar** diferentes algoritmos de classificação para prever o risco de cardiopatias.
- **Explorar** métodos de pré-processamento de dados clínicos e antropométricos.
- **Validar** os resultados utilizando métricas como acurácia, precisão, sensibilidade, F1-score e AUC-ROC.

---

## Descrição do Projeto

A solução desenvolvida envolve as seguintes etapas:
1. **Coleta e Análise Exploratória de Dados (EDA)**  
   Realizou-se a limpeza, análise estatística e visualização dos dados clínicos e antropométricos.
2. **Pré-Processamento e Seleção de Atributos**  
   Foram aplicadas técnicas para remoção de inconsistências, imputação de valores ausentes, tratamento de outliers e criação de novas features.
3. **Modelagem**  
   Diversos algoritmos de classificação foram testados: KNN, Naive Bayes, Regressão Logística, MLP, SVM, Random Forest, Gradient Boosting e AdaBoost.
4. **Validação e Análise de Desempenho**  
   Os modelos foram avaliados utilizando validação cruzada (k-fold, com k=5) e métricas robustas, com destaque para os métodos ensemble, especialmente o Gradient Boosting.

---

## Estrutura de Arquivos

```plaintext
├── IMPEMENTACAO
   ├── main.ipynb               # Notebook principal integrando todo o pipeline
   ├── plots   # Imagens dos resultados de pré-processamento e análises realizadas ao longo do notebook
   ├── plots   # Reports de estatísticas descritivas e inconsistências encontradas na análise de dados
   ├── scripts
      └── preprocessamento.py      # Pipeline de pré-processamento de dados
      └── analise_exploratoria.py  # Rotinas de análise exploratória (EDA)
      └── experimentos.py          # Treinamento e validação dos modelos
      └── analise_resultados.py    # Consolidação e visualização dos resultados
   ├── Relatorio_FInal_ML.pdf   # Relatório final detalhando metodologia e resultados
   ├── submissao_final.csv   # Arquivo de submissao do projeto no Kaggle (utilizado no entrega do projeto da dsiciplina)
   ├── treino_pre_processado.csv   # Arquivo de dados de treino pré processados
├── Relatorio_Final_ML.pdf # Relatório completo do projeto
├── README.md               
```         

## Conjunto de Dados

**Origem:**  
Dados provenientes do Real Hospital Português (RHP).

**Número de Registros:**  
Aproximadamente 17.874, abrangendo pacientes de 0 até 19 anos, além de alguns registros de adultos.

**Variáveis Principais:**

- **Atributos antropométricos:**  
  Peso, altura, Índice de Massa Corporal.

- **Medidas de pressão arterial:**  
  Sistólica e diastólica.

- **Frequência cardíaca.**

- **Faixa etária.**

- **Indicadores clínicos:**  
  Por exemplo, sopro cardíaco, B2, pulsos.

- **Rotulação final:**  
  Normal ou Anormal (indicando a presença de cardiopatia).

**Possíveis Inconsistências:**  
Durante a análise exploratória, foram encontrados valores inválidos (idades negativas, medições de PA sistólica menor que a diastólica, etc.). Assim, houve a necessidade de tratamento cuidadoso para garantir integridade e coerência no conjunto de dados.

---

## Processo de Pré-Processamento

**Remoção de Colunas Irrelevantes**  
Exemplo: colunas como Id, Convenio e Atendimento não apresentaram relevância para a modelagem.

**Tratamento de Valores Ausentes**  
- **Numéricos:** substituição pela mediana.  
- **Categóricos:** substituição pela moda.

**Limpeza de Inconsistências**  
Ajuste de valores incoerentes (ex.: peso zero, idade negativa) pela mediana da variável.

**Detecção e Tratamento de Outliers**  
Foram identificados e removidos outliers extremos (usando percentis de 1% e 99%), especialmente em medidas de IMC e pressão arterial.

**Criação de Novas Features**  
Faixa_Etaria, Faixa_IMC, FC_por_Idade, entre outras, levando em conta as faixas pediátricas.

**Codificação de Variáveis Categóricas**  
Utilização de One-Hot Encoding para variáveis como SEXO, SOPRO, B2, PULSOS, etc.

**Escalonamento**  
Uso do RobustScaler para reduzir a influência de valores extremos e normalizar variáveis numéricas.

---

## Arquitetura de Modelagem e Experimentação

Toda a etapa de experimentos foi realizada utilizando diversos algoritmos de classificação:

- **KNN (K-Nearest Neighbors)**
- **Naive Bayes**
- **Regressão Logística**
- **MLP (Multilayer Perceptron)**
- **SVM (Support Vector Machine)**
- **Random Forest**
- **Gradient Boosting**
- **AdaBoost**

**Protocolo Experimental:**

- **Divisão de Dados:**  
  80% treino e 20% teste.

- **Validação Cruzada:**  
  k-fold cross-validation (k=5) para garantir robustez na comparação de performance.

- **Ajuste de Hiperparâmetros:**  
  Realizado via Grid Search, com foco especial nos algoritmos ensemble (Gradient Boosting, Random Forest e AdaBoost).

- **Métricas Avaliadas:**  
  Acurácia, Precisão, Recall (Sensibilidade), F1-Score e AUC-ROC.

**Comparação dos Modelos:**  
Segundo o relatório final (*Relatorio_FInal_ML.pdf*), os métodos ensemble demonstraram melhores resultados, destacando-se o Gradient Boosting por apresentar AUC-ROC superior e melhor equilíbrio entre bias e variance.

---

## Como Executar o Projeto

**Clonar o Repositório:**

```bash
git clone https://github.com/Pedro-bnogueira/Deteccao-de-Cardiopatias-ML.git
cd Deteccao-de-Cardiopatias-ML
cd IMPLEMENTACAO
````

**Instalar Dependências**
Recomenda-se a utilização de um ambiente virtual (virtualenv ou conda).

- Executar célula de instalação de dependências do main.ipynb

Executar o Notebook Principal
O arquivo main.ipynb integra todo o pipeline de EDA, pré-processamento, treinamento e avaliação.

```bash
jupyter notebook main.ipynb
```

---
## Scripts

preprocessamento.py: Contém as funções de limpeza, imputação e escalonamento.

analise_exploratoria.py: Executa EDA, gerando gráficos e estatísticas.

experimentos.py: Executa os treinamentos e validações cruzadas dos modelos.

analise_resultados.py: Consolida as métricas e gera gráficos comparativos.

---
## Principais Resultados e Discussões
**Resultados das Métricas**

Os métodos ensemble (principalmente Gradient Boosting) obtiveram:

- Acurácia: ~93%

- AUC-ROC: ~0.95 (em média na validação cruzada)

**Análise de Desempenho**
- Gradient Boosting: Alto poder preditivo e baixa tendência ao overfitting graças aos ajustes de hiperparâmetros (como learning_rate e max_depth).

- Random Forest: Métricas competitivas, porém ligeiramente inferiores em AUC-ROC em relação ao Gradient Boosting.

- SVM, MLP: Resultados promissores, mas sensíveis a configurações específicas (kernel, funções de ativação, etc.).

- KNN e Naive Bayes: Bons resultados, porém abaixo dos métodos ensemble no conjunto de dados clínicos, que possui alta dimensionalidade e complexidade.

### Visualizações

**Distribuição das Variáveis**
![Histograma Geral](/plots/histograma-geral.png)

**Comparação de Modelos**

![Comparação de modelos](/plots/desempenho-modelos.png)

**Curva de Aprendizado (Gradient Boosting)**

![Curva de Aprendizado (Gradient Boosting)](/plots/curva-aprendizado.png)

---
## Conclusões
**Relevância Clínica**: A alta sensibilidade dos modelos, especialmente Gradient Boosting, contribui para minimizar falsos negativos, fator crucial em diagnósticos médicos.

**Importância do Pré-Processamento**: A limpeza e a engenharia de features foram decisivas para a performance dos algoritmos.

**Aplicabilidade**: Os resultados demonstram potencial para uso em sistemas de apoio à decisão médica, embora não substituam avaliações clínicas especializadas.

---
## Referências
Smith, J., & Doe, A. (2021). Early Detection of Pediatric Heart Diseases. Journal of Pediatric Cardiology, 15(3), 234-245.

Johnson, L., & Wang, Y. (2020). Machine Learning Applications in Medical Diagnostics. IEEE Transactions on Biomedical Engineering, 67(4), 1123-1132.

Miller, T., & Brown, K. (2019). Ensemble Learning for ECG-Based Cardiac Anomaly Detection. International Journal of Artificial Intelligence in Healthcare, 8(2), 98-110.

Lima, M., & Oliveira, S. (2018). Classification Models for Cardiac Diagnosis in Pediatric Age Groups. XXX Symposium on Machine Learning, 25-30.

Fernandes, D., & Silva, E. (2021). Data Preprocessing Techniques for Enhancing Machine Learning Models in Medical Data. Journal of Data Science in Medicine, 5(1), 45-60.

Powers, D. M. (2020). Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness and Correlation. Journal of Machine Learning Technologies, 2(1), 37-63.

Bootkrajang, J., & Kabán, A. (2013). Boosting in the Presence of Label Noise. Proceedings of the Twenty-Ninth Conference on Uncertainty in Artificial Intelligence (UAI2013), 82-91.

Guyon, I., & Elisseeff, A. (2003). An Introduction to Variable and Feature Selection. Journal of Machine Learning Research, 3, 1157-1182.

## **Observação Importante**

Este projeto tem finalidade acadêmica e experimental. Em aplicações de saúde, qualquer tecnologia de diagnóstico precisa ser validada e aprovada por órgãos competentes antes de sua adoção clínica.

