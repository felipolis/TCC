import os
import cv2
import numpy as np
from joblib import Parallel, delayed

# Diretório onde estão os espectrogramas coloridos
spectrogram_dir = "../data/espectogramas"

# Diretório para salvar os patches
patch_dir = "../data/patches"

# Tamanho da janela (número de colunas) para cortar os espectrogramas
n_columns = 300  # Altere para o número desejado

# Se o diretório para salvar os patches não existir, crie-o
if not os.path.exists(patch_dir):
    os.mkdir(patch_dir)

# Função para cortar e salvar os patches de um espectrograma
def cut_and_save_patches(spectrogram_path, output_dir, n_columns):
    # Carrega o espectrograma colorido
    spectrogram = cv2.imread(spectrogram_path)

    # Calcula a largura (número de colunas) do espectrograma
    width = spectrogram.shape[1]

    # Calcula quantos patches podem ser criados
    num_patches = width // n_columns

    # Verifica se tem resto
    if width % n_columns != 0:
        num_patches += 1

    # Loop para cortar e salvar os patches
    for i in range(num_patches):
        start_col = i * n_columns
        end_col = start_col + n_columns

        # Corta o patch
        patch = spectrogram[:, start_col:end_col, :]

        # Se for o último patch e não for do mesmo tamanho que os outros, preenche com zeros
        if i == num_patches - 1 and patch.shape[1] < n_columns:
            padding = n_columns - patch.shape[1]
            patch = np.pad(patch, ((0, 0), (0, padding), (0, 0)), mode='constant')

        # Cria o nome do arquivo de saída
        output_path = os.path.join(output_dir, f"{os.path.basename(spectrogram_path).replace('.png', f'_patch_{i}.png')}")

        # Salva o patch
        cv2.imwrite(output_path, patch)

def process_spectrogram(spectrogram_path):
    output_subdir = os.path.relpath(os.path.dirname(spectrogram_path), spectrogram_dir)
    output_dir = os.path.join(patch_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    cut_and_save_patches(spectrogram_path, output_dir, n_columns)


# Percorre os espectrogramas e corta em patches
Parallel(n_jobs=4)(delayed(process_spectrogram)(os.path.join(root, file)) for root, dirs, files in os.walk(spectrogram_dir) for file in files if file.endswith(".png"))

print(f"Patches cortados e salvos em '{patch_dir}'")
