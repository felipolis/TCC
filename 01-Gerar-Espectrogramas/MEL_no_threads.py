import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Caminho dos diretórios de áudio e espectrogramas
audio_dir = "../data/train_audio/"
spectrogram_dir = "../data/espectogramas"

# Se não existir o diretorio ./train_espectogramas, cria
if not os.path.exists(spectrogram_dir):
    os.mkdir(spectrogram_dir)

# Função para criar espectrograma e salvar a imagem
def create_spectrogram(audio_path, output_path):
    # Carrega o arquivo de áudio utilizando a biblioteca librosa
    y, sr = librosa.load(audio_path, sr=None)

    # calcula o mel espectrograma do áudio com n_fft=2048, hop_length=1024
    D = librosa.amplitude_to_db(np.abs(librosa.feature.melspectrogram(y=y, n_fft=2048, hop_length=1024)), ref=np.max)
    
    # Cria a figura com o tamanho proporcional
    try:
        # Cria a figura com o tamanho proporcional
        fig = plt.figure(figsize=(D.shape[1]/72, D.shape[0]/72), dpi=72)

        # Plota o espectrograma
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        plt.pcolormesh(D)

        # Salva a figura no diretório de espectrogramas
        plt.savefig(output_path, dpi=72, pad_inches=0, bbox_inches='tight')

        # Fecha a figura para liberar recursos de memória
        plt.close()

        return True
    
    except(ValueError):
        
        print("******************************************************")
        print("Erro ao criar espectrograma para o arquivo: " + audio_path)
        print("Número de amostras no arquivo: " + str(len(y)))
        print("******************************************************")

        return False

espec_created = 0
espec_not_created = 0

# Percorrendo os diretórios de áudio
for key in os.listdir(audio_dir):
    key_dir = os.path.join(audio_dir, key)
    if os.path.isdir(key_dir):
        spectrogram_key_dir = os.path.join(spectrogram_dir, key)
        os.makedirs(spectrogram_key_dir, exist_ok=True)
        
        for audio_name in os.listdir(key_dir):
            if audio_name.endswith(".ogg"):
                audio_path = os.path.join(key_dir, audio_name)
                spectrogram_path = os.path.join(spectrogram_key_dir, audio_name.replace(".ogg", ".png"))
                created = create_spectrogram(audio_path, spectrogram_path)
                if created:
                    espec_created += 1
                else:
                    espec_not_created += 1
                print(f"Espectrograma criado para {audio_name}")

print(f"Espectrogramas criados: {espec_created}")
print(f"Espectrogramas não criados: {espec_not_created}")
