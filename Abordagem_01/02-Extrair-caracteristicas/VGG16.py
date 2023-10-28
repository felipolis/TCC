import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing import image
from skimage import feature
from tensorflow.keras.applications.vgg16 import preprocess_input

# Diretórios
spectrogram_dir = "../../data/espectrogramas"
diretorio_caracteristicas = '../../data/caracteristicas/'
arquivo_csv = '../../data/caracteristicas/VGG16.csv'

# Constantes
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

# Se o diretório não existir, cria
if not os.path.exists(diretorio_caracteristicas):
    os.makedirs(diretorio_caracteristicas)

# Carregue o modelo VGG16 com os parâmetros desejados
vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

def extrair_caracteristicas_VGG16(imagem):
    # Redimensione a imagem para o tamanho esperado pelo modelo VGG16
    imagem = cv2.resize(imagem, (224, 224))
    # Pré-processamento da imagem de acordo com as especificações da VGG16
    imagem = preprocess_input(imagem)
    # Expanda as dimensões da imagem para corresponder ao input_shape do modelo
    imagem = np.expand_dims(imagem, axis=0)
    # Obtenha as características usando o modelo VGG16
    features = vgg_model.predict(imagem)
    # Aplique flatten nas características
    features = features.flatten()
    return features

# Inicialize um DataFrame vazio para armazenar as características VGG16
df = pd.DataFrame()

# Percorra os diretórios da lista de classes
for key in os.listdir(spectrogram_dir):
    if key in classes:
        key_path = os.path.join(spectrogram_dir, key)

        # Percorra todas as imagens no diretório
        for image_name in os.listdir(key_path):
            image_path = os.path.join(key_path, image_name)
            # Carregue a imagem
            imagem = cv2.imread(image_path)
            # Extraia as características da imagem
            features = extrair_caracteristicas_VGG16(imagem)
            # Obtenha o nome do arquivo original
            nome_arquivo_original = image_name.split('.')[0]
            # Adicione as características ao DataFrame
            data_to_append = pd.Series([key, nome_arquivo_original] + list(features))
            df = pd.concat([df, data_to_append.to_frame().T], ignore_index=True)

            print(f"Processando {image_path}")

# Defina os nomes das colunas no DataFrame
colunas = ['key', 'nome_arquivo'] + [f'LBP_{i}' for i in range(len(features))]
df.columns = colunas

# Salve o DataFrame em um arquivo CSV
df.to_csv(arquivo_csv, index=False)