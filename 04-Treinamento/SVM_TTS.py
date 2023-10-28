import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

############## SEPARANDO ENTRE DADOS DE TREINO E TESTE ##############

# Pega as duas primeiras colunas (key, nome_arquivo) do arquivo LBP.csv
df = pd.read_csv('../data/caracteristicas/LBP.csv', usecols=[0,1])

# Remove as linhas onde há um nome_arquivo repetido
df = df.drop_duplicates(subset=['nome_arquivo'])

# Remova o elemento em que só há uma ocorrência de classe
df = df.groupby('key').filter(lambda x: len(x) > 1)

# Coloca a coluna key em uma lista chamada classes e a coluna nome_arquivo em uma lista chamada arquivos
classes = df['key'].tolist()
arquivos = df['nome_arquivo'].tolist()

# Separe quais arquivos serão usados para treino e quais serão usados para teste
arquivos_treino, arquivos_teste, classes_treino, classes_teste = train_test_split(arquivos, classes, test_size=0.2, random_state=42, stratify=classes)

# Carregar o arquivo LBP.csv
df_lbp = pd.read_csv('../data/caracteristicas/LBP.csv')

# Selecionar apenas as linhas que estão nos arquivos de treino
df_lbp_treino = df_lbp[df_lbp['nome_arquivo'].isin(arquivos_treino)]

# Selecionar apenas as linhas que estão nos arquivos de teste
df_lbp_teste = df_lbp[df_lbp['nome_arquivo'].isin(arquivos_teste)]

# Separa entre y_treino e X_treino
y_treino = df_lbp_treino['key'].values
X_treino = df_lbp_treino.drop(['key', 'nome_arquivo'], axis=1).values

# Separa entre y_teste e X_teste
y_teste = df_lbp_teste['key'].values
X_teste = df_lbp_teste.drop(['key', 'nome_arquivo'], axis=1).values

############## NORMALIZANDO OS DADOS ##############

# Cria um objeto para normalizar os dados
scaler = StandardScaler()

# Normaliza os dados de treino
X_treino = scaler.fit_transform(X_treino)

# Normaliza os dados de teste
X_teste = scaler.transform(X_teste)

############## TRATANDO OS VALORES NAN ##############

# Substitui os valores NaN por 0
X_treino = np.nan_to_num(X_treino)
X_teste = np.nan_to_num(X_teste)

############## ENCONTRA OS MELHORES PARÂMETROS SVM ##############

# Cria um objeto do tipo SVM
svm = SVC()

# Define os valores que serão testados para o hiperparâmetro C
C_range = list(range(1, 20, 2))

# Define os valores que serão testados para o gamma
gamma_range = list(np.logspace(-4, 1, 6, base=2)) + ['scale', 'auto']

# Define os valores que serão testados para o hiperparâmetro kernel
kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']

# Cria um dicionário com os valores dos hiperparâmetros
parametros_svm = dict(C=C_range, kernel=kernel_options, gamma=gamma_range)

# Cria um objeto do tipo StratifiedKFold
skf = StratifiedKFold(n_splits=5)

# Cria um objeto do tipo GridSearchCV
grid_svm = GridSearchCV(svm, parametros_svm, cv=skf, scoring='accuracy', verbose=1, n_jobs=-1)

# Treina o modelo
grid_svm.fit(X_treino, y_treino)

# Imprime os melhores parâmetros
print(f'Melhor valor para C: {grid_svm.best_params_["C"]}')
print(f'Melhor valor para kernel: {grid_svm.best_params_["kernel"]}')
print(f'Melhor valor para gamma: {grid_svm.best_params_["gamma"]}')
print(f'Melhor Acurácia: {grid_svm.best_score_}')

############## TREINANDO O MODELO SVM COM OS MELHORES PARÂMETROS ##############

# Cria um objeto do tipo SVM com os melhores parâmetros
svm = SVC(C=grid_svm.best_params_["C"], kernel=grid_svm.best_params_["kernel"], gamma=grid_svm.best_params_["gamma"])

# Treina o modelo
svm.fit(X_treino, y_treino)

# Testa o modelo
y_pred = svm.predict(X_teste)

############## VOTAÇÃO ##############

# copie o df df_lbp_teste para df_lbp_teste_votacao e adicione a coluna 'predicao'
df_lbp_teste_votacao = df_lbp_teste.copy()
df_lbp_teste_votacao['predicao'] = y_pred

# Faça a votação para cada nome_arquivo
df_lbp_teste_votacao = df_lbp_teste_votacao.groupby('nome_arquivo')['predicao'].agg(lambda x:x.value_counts().index[0]).reset_index()

# Crie um df com as classes reais
df_lbp_teste_real = df_lbp_teste.drop_duplicates(subset=['nome_arquivo'])

# Junte os dois df
df_lbp_teste_real = df_lbp_teste_real.merge(df_lbp_teste_votacao, on='nome_arquivo')

############## AVALIANDO O MODELO ##############

# Imprime o relatório de classificação
print(classification_report(df_lbp_teste_real['key'], df_lbp_teste_real['predicao']))
