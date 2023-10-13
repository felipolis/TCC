import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

audio_path = '../data/train_audio/thrnig1/XC25659.ogg'
output_path = './XC25659.png'

def create_spectrogram(audio_path, output_path):
    # Carrega o arquivo de áudio utilizando a biblioteca librosa
    y, sr = librosa.load(audio_path, sr=None)

    # Verifica o comprimento do áudio em segundos
    audio_length = librosa.get_duration(y=y, sr=sr)

    # Define o limite de 10 minutos em segundos
    max_duration = 600  # 10 minutos * 60 segundos

    if audio_length > max_duration:
        # Se o áudio for maior que 10 minutos, corte-o
        y = y[:int(max_duration * sr)]

    # Calcula a Transformada de Fourier de Curto Prazo (STFT) do sinal de áudio com n_fft=2048, hop_length=1024
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
    
    except:
        
        print("******************************************************")
        print("Erro ao criar espectrograma para o arquivo: " + audio_path)
        print("Número de amostras no arquivo: " + str(len(y)))
        print("******************************************************")

        return False
    
create_spectrogram(audio_path, output_path)