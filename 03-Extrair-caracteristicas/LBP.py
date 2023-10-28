import os
import cv2
import pandas as pd
from skimage import feature
import numpy as np
import matplotlib.pyplot as plt

# Diretórios
diretorio_imagens = '../data/patches/'
diretorio_caracteristicas = '../data/caracteristicas/'
arquivo_csv = '../data/caracteristicas/LBP.csv'
spectrogram_dir = "../data/espectogramas"

# Parâmetros LBP
QUANT_CLASSES = 3
POINTS = 8
RADIUS = 2

# Encontre as classes mais comuns
class_names = []
class_count = {}

for root, dirs, files in os.walk(spectrogram_dir):
    for dir in dirs:
        class_names.append(dir)

for root, dirs, files in os.walk(spectrogram_dir):
    for dir in dirs:
        class_count[dir] = len(os.listdir(os.path.join(root, dir)))

class_count = dict(sorted(class_count.items(), key=lambda item: item[1], reverse=True))
classes = list(class_count.keys())[:QUANT_CLASSES]
print(f"Classes com mais imagens: {classes}")

# Crie o diretório se não existir
if not os.path.exists(diretorio_caracteristicas):
    os.makedirs(diretorio_caracteristicas)

def calcular_LBP(imagem):
    # Calcula o vetor de características LBP
    lbp = feature.local_binary_pattern(imagem, POINTS, RADIUS, method="nri_uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

    # Normaliza o histograma
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return hist

# Inicialize um DataFrame vazio para armazenar as características LBP
df = pd.DataFrame()

# Percorra os diretórios da lista de classes
for key in os.listdir(diretorio_imagens):
    if key in classes:
        key_path = os.path.join(diretorio_imagens, key)

        # Percorra todas as imagens no diretório
        for image_name in os.listdir(key_path):
            image_path = os.path.join(key_path, image_name)

            # Carregue a imagem
            image = cv2.imread(image_path)

            # Converta para escala de cinza
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calcule o vetor de características LBP
            caracteristicas_lbp = calcular_LBP(image)

            # Obtenha o nome do arquivo original
            nome_arquivo_original = image_name.split('_')[0]
            
            # Adicione as características e o nome do arquivo ao DataFrame
            data_to_append = pd.Series([key, nome_arquivo_original] + list(caracteristicas_lbp))
            df = pd.concat([df, data_to_append.to_frame().T], ignore_index=True)

            print(f'Processando {image_path}')

# Defina os nomes das colunas no DataFrame
colunas = ['key', 'nome_arquivo'] + [f'LBP_{i}' for i in range(len(caracteristicas_lbp))]
df.columns = colunas

# Salve o DataFrame em um arquivo CSV
df.to_csv(arquivo_csv, index=False)
