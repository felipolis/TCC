import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from scipy.spatial import distance

def calcular_ponto_maior_densidade(dados_classe):
    # Remove as linhas com valores ausentes
    dados_classe = dados_classe.dropna(subset=['longitude', 'latitude'])

    # Cria uma matriz 2D com os valores de longitude e latitude
    data = dados_classe[['longitude', 'latitude']].values

    if len(data) == 0:
        return None  # Retorna None se não houver pontos de dados válidos

    # Aplica o algoritmo KMeans para encontrar os clusters
    kmeans = KMeans(n_clusters=1)  # Para encontrar um único ponto de maior densidade
    kmeans.fit(data)

    # O ponto central do cluster representa o ponto de maior densidade
    ponto_maior_densidade = kmeans.cluster_centers_[0]

    return ponto_maior_densidade

def calcular_ponto_referencia(data):
    # Pega as coordenadas de cada classe
    coordenadas = [coordenadas for classe, coordenadas in data.items()]

    # Aplica o algoritmo KMeans para encontrar um ponto de referência entre os pontos de maior densidade
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(coordenadas)

    # O ponto central do cluster representa o ponto de referência
    ponto_referencia = kmeans.cluster_centers_[0]

    return ponto_referencia

def encontrar_classes_proximas(arquivo_csv, numero_minimo_amostras, N):
    # Ler o arquivo CSV
    data = pd.read_csv(arquivo_csv)

    # Contar o número de amostras por classe
    class_counts = data['primary_label'].value_counts()

    # Selecionar classes com mais de 'numero_minimo_amostras' amostras
    selected_classes = class_counts[class_counts > numero_minimo_amostras].index.tolist()

    # Filtrar os dados para classes selecionadas
    filtered_data = data[data['primary_label'].isin(selected_classes)]

    # Agrupar por classe
    grouped = filtered_data.groupby('primary_label')

    # Encontrar os pontos de maior densidade para cada classe usando KMeans
    highest_density_points = {}
    for class_name, class_data in grouped:
        highest_density_point = calcular_ponto_maior_densidade(class_data)
        highest_density_points[class_name] = highest_density_point

    # Encontrar o ponto de referência entre os pontos de maior densidade usando KMeans
    reference_point = calcular_ponto_referencia(highest_density_points)

    # Ordenar os pontos por distância para o ponto de referência
    distances = {class_name: distance.euclidean(reference_point, point) for class_name, point in highest_density_points.items()}
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])

    # Selecionar as N classes mais próximas do ponto de referência
    closest_classes = [class_name for class_name, dist in sorted_distances[:N]]

    # Mapa
    fig, ax = plt.subplots()
    # Create a color map
    colors = list(mcolors.CSS4_COLORS.keys())
    # Plot each class point with a unique color
    for index, (class_name, highest_density_point) in enumerate(highest_density_points.items()):
        ax.scatter(highest_density_point[0], highest_density_point[1], color=colors[index % len(colors)], label=class_name)
    # Plot the reference point
    ax.scatter(reference_point[0], reference_point[1], color='red', label='Reference Point')
    ax.legend()
    plt.show()

    return closest_classes

# Exemplo de uso da função
classes_proximas = encontrar_classes_proximas('./data/metadata.csv', 100, 10)
print("Classes mais próximas com mais de 100 amostras:", classes_proximas)
