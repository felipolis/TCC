import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

############## SEPARANDO ENTRE DADOS DE TREINO E TESTE ##############

# Pega as duas primeiras colunas (key, nome_arqivo) do arquivo LBP.csv
df = pd.read_csv('../data/caracteristicas/ResNet50.csv', usecols=[0,1])

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
df_lbp = pd.read_csv('../data/caracteristicas/ResNet50.csv')

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

############## ENCONTRA OS MELHORES PARAMETROS KNN ##############

# Cria um objeto do tipo KNN
knn = KNeighborsClassifier()

# Define os valores que serão testados para cada parâmetro
k_range = list(range(1, 20, 2))

# Cria um dicionário com os valores dos parâmetros
parametros_knn = dict(n_neighbors=k_range)

# Cria um objeto do tipo StratifiedKFold
skf = StratifiedKFold(n_splits=5)

# Cria um objeto do tipo GridSearchCV
grid_knn = GridSearchCV(knn, parametros_knn, cv=skf, scoring='accuracy', verbose=1, n_jobs=-1)

# Treina o modelo
grid_knn.fit(X_treino, y_treino)

# Imprime os melhores parâmetros
print(f'Melhor valor para n_neighbors: {grid_knn.best_params_["n_neighbors"]}')
print(f'Melhor Acurácia: {grid_knn.best_score_}')

############## TREINANDO O MODELO KNN COM OS MELHORES PARAMETROS ##############

# Cria um objeto do tipo KNN
knn = KNeighborsClassifier(n_neighbors=grid_knn.best_params_["n_neighbors"])

# Treina o modelo
knn.fit(X_treino, y_treino)

# Testa o modelo
y_pred = knn.predict(X_teste)

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
