import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor
from skimage import io

# Diretórios
diretorio_imagens = '../data/patches/'
diretorio_caracteristicas = '../data/caracteristicas/'
arquivo_csv = '../data/caracteristicas/Gabor.csv'
spectrogram_dir = "../data/espectogramas"

# Parâmetros Gabor
FREQUENCIES = [0.1, 0.3, 0.5]
THETAS = [0, np.pi/4, np.pi/2, 3*np.pi/4]
QUANT_CLASSES = 2

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

def calcular_Gabor(imagem):
    # Calcula o vetor de características Gabor
    features = []
    for freq in FREQUENCIES:
        for theta in THETAS:
            filtered_image, _ = gabor(imagem, frequency=freq, theta=theta)
            features.extend(filtered_image.ravel())
    return features

# Inicialize um DataFrame vazio para armazenar as características Gabor
df = pd.DataFrame()

# Percorra os diretórios da lista de classes
for key in os.listdir(diretorio_imagens):
    if key in classes:
        key_path = os.path.join(diretorio_imagens, key)

        # Percorra todas as imagens no diretório
        for image_name in os.listdir(key_path):
            image_path = os.path.join(key_path, image_name)

            # Carregue a imagem
            image = io.imread(image_path, as_gray=True)

            # Calcule o vetor de características Gabor
            caracteristicas_gabor = calcular_Gabor(image)

            # Obtenha o nome do arquivo original
            nome_arquivo_original = image_name.split('_')[0]

            # Adicione as características e o nome do arquivo ao DataFrame
            data_to_append = pd.Series([key, nome_arquivo_original] + caracteristicas_gabor)
            df = pd.concat([df, data_to_append.to_frame().T], ignore_index=True)

            print(f'Processando {image_path}')

# Defina os nomes das colunas no DataFrame
colunas = ['key', 'nome_arquivo'] + [f'Gabor_{i}' for i in range(len(caracteristicas_gabor))]
df.columns = colunas

# Salve o DataFrame em um arquivo CSV
df.to_csv(arquivo_csv, index=False)
