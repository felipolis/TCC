import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

############## SEPARANDO ENTRE DADOS DE TREINO E TESTE ##############

# Pega as duas primeiras colunas (key, nome_arquivo) do arquivo LBP.csv
df = pd.read_csv('../data/caracteristicas/LBP.csv', usecols=[0, 1])

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
    # Separe quais arquivos serão usados para treino e quais serão usados para teste
    arquivos_treino, arquivos_teste = np.array(arquivos)[train_index], np.array(arquivos)[test_index]
    classes_treino, classes_teste = np.array(classes)[train_index], np.array(classes)[test_index]

    # Carregar o arquivo LBP.csv
    df_lbp = pd.read_csv('../data/caracteristicas/LBP.csv')

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

    # Tratando os valores NaN
    x_treino = np.nan_to_num(x_treino)
    x_teste = np.nan_to_num(x_teste)

    # Encontra os melhores parâmetros para o SVM
    svm = SVC()
    C_range = [0.01, 0.1, 1, 10]
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
    gamma_range = list(np.logspace(-4, 1, 6, base=2)) + ['scale', 'auto']
    parametros_svm = dict(C=C_range, kernel=kernel_options, gamma=gamma_range)
    grid_svm = GridSearchCV(svm, parametros_svm, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

    grid_svm.fit(x_treino, y_treino)

    print(f'Melhor valor de C: {grid_svm.best_params_["C"]}')
    print(f'Melhor valor de kernel: {grid_svm.best_params_["kernel"]}')
    print(f'Melhor valor de gamma: {grid_svm.best_params_["gamma"]}')
    print(f'Melhor Acurácia: {grid_svm.best_score_}')

    # Treina o SVM com os melhores parâmetros
    svm = SVC(C=grid_svm.best_params_["C"], kernel=grid_svm.best_params_["kernel"], gamma=grid_svm.best_params_["gamma"])
    svm.fit(x_treino, y_treino)

    # Faz a predição
    y_pred = svm.predict(x_teste)

    # Calcula a acurácia
    acc.append(accuracy_score(y_teste, y_pred))

    # Calcula a F1-Score
    f1s.append(f1_score(y_teste, y_pred, average='weighted'))

print(f'Acurácia: {np.mean(acc)}')
print(f'F1-Score: {np.mean(f1s)}')
