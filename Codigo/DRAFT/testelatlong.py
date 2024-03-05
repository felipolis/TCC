import pandas as pd
import numpy as np

def extrair_lat_long(metadata_csv, filename):
    # Extrair a classe do nome do arquivo
    classe = filename.split('/')[0]

    # Carregar o arquivo CSV de metadados
    metadata = pd.read_csv(metadata_csv)

    # Encontrar os registros correspondentes no metadata.csv com base na classe e no nome do arquivo
    metadata_subset = metadata[(metadata['primary_label'] == classe) & (metadata['filename'].str.contains(filename))]

    latitude = metadata_subset['latitude'].values[0]
    longitude = metadata_subset['longitude'].values[0]

    # Verifica se latitude e longitude são válidas
    if np.isnan(latitude) or np.isnan(longitude):
        media_latitude = metadata[metadata['primary_label'] == classe]['latitude'].mean()
        media_longitude = metadata[metadata['primary_label'] == classe]['longitude'].mean()

        # 4 casas depois da vírgula
        latitude = round(media_latitude, 4)
        longitude = round(media_longitude, 4)

        return latitude, longitude
    
    return latitude, longitude


# Arquivo CSV de entrada e arquivo para salvar as informações
input_file = '../data/metadata.csv'  # Substitua pelo seu arquivo CSV

# chama a função
latitude, longitude = extrair_lat_long(input_file, 'abethr1/XC585802.ogg')

