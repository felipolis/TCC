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

# Cria o objeto skf que divide os dados em 10 partes
skf = StratifiedKFold(n_splits=10)
acc = []
f1s = []

# Divide os dados em 10 partes
for train_index, test_index in skf.split(arquivos, classes):
    #Separe quais arquivos serão usados para treino e quais serão usados para teste
    arquivos_treino, arquivos_teste = np.array(arquivos)[train_index], np.array(arquivos)[test_index]
    classes_treino, classes_teste = np.array(classes)[train_index], np.array(classes)[test_index]

    # Carregar o arquivo LBP.csv
    df_lbp = pd.read_csv('../data/caracteristicas/ResNet50.csv')

    # Selecionar apenas as linhas que estão nos arquivos de treino
    df_lbp_treino = df_lbp[df_lbp['nome_arquivo'].isin(arquivos_treino)]

    # Selecionar apenas as linhas que estão nos arquivos de teste
    df_lbp_teste = df_lbp[df_lbp['nome_arquivo'].isin(arquivos_teste)]

    # Separa entre y_treino e x_treino
    y_treino = df_lbp_treino['key']
    x_treino = df_lbp_treino.drop(['key', 'nome_arquivo'], axis=1)

    # Separa entre y_teste e x_teste
    y_teste = df_lbp_teste['key']
    x_teste = df_lbp_teste.drop(['key', 'nome_arquivo'], axis=1)

    # Normaliza os dados
    scaler = StandardScaler()
    scaler.fit(x_treino)
    x_treino = scaler.transform(x_treino)
    x_teste = scaler.transform(x_teste)

    # tratando so valores NaN
    x_treino = np.nan_to_num(x_treino)
    x_teste = np.nan_to_num(x_teste)

    # Encontra os melhores parâmetros para o KNN
    knn = KNeighborsClassifier()
    k_range = list(range(1, 20, 2))
    parametros_knn = dict(n_neighbors=k_range)
    skf = StratifiedKFold(n_splits=5)
    grid_knn = GridSearchCV(knn, parametros_knn, cv=skf, scoring='accuracy', verbose=1, n_jobs=-1)

    grid_knn.fit(x_treino, y_treino)

    print(f'Melhor valor de K: {grid_knn.best_params_["n_neighbors"]}')
    print(f'Melhor valor de acurácia: {grid_knn.best_score_}')

    # Treina o KNN com os melhores parâmetros
    knn = KNeighborsClassifier(n_neighbors=grid_knn.best_params_["n_neighbors"])
    knn.fit(x_treino, y_treino)

    # Faz a predição
    y_pred = knn.predict(x_teste)

    # Votação
    df_lbp_teste_votacao = df_lbp_teste.copy()
    df_lbp_teste_votacao['predicao'] = y_pred
    df_lbp_teste_votacao = df_lbp_teste_votacao.groupby('nome_arquivo')['predicao'].agg(lambda x:x.value_counts().index[0]).reset_index()

    df_lbp_teste_real = df_lbp_teste.drop_duplicates(subset=['nome_arquivo'])
    df_lbp_teste_real = df_lbp_teste_real.merge(df_lbp_teste_votacao, on='nome_arquivo')

    # Calcula a acurácia
    acc.append(accuracy_score(df_lbp_teste_real['key'], df_lbp_teste_real['predicao']))

    # Calcula a f1-score
    f1s.append(classification_report(df_lbp_teste_real['key'], df_lbp_teste_real['predicao'], output_dict=True)['weighted avg']['f1-score'])

print(f'Acurácia: {np.mean(acc)}')
print(f'F1-Score: {np.mean(f1s)}')
    
