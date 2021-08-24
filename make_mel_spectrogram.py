# 对wav音频文件进行预处理，转换为梅尔频谱图
# 进行高频率波，从中提取出鸟类鸣叫的高频信息
# 是否需要对每个文件进行时长以10s为单位切分，以填充训练集数量
# 暂时我们每个文件仅取 10s的长度进行频谱图转换


import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np
import noisereduce as no
import cv2

sound_file = 'B000_0194.wav'    # 11s
sound_file = 'B099_0393.wav'    # 9s
sound_file = 'B000_0017.wav'    # 1mins
signal, sampling_rate = librosa.load(path=sound_file, duration=10, sr=32000)    # 默认 sr=22050
sr = sampling_rate
# print(sound_sample)
# signal = no.reduce_noise(signal, signal, verbose=False)
print(signal.shape)
print(signal.max()) # 1.0030551
print(signal.min()) # -1.0012032

# 展示采样后的音频波形图 recording signal,其中低频都是环境噪音，高频是鸟鸣声
plt.figure(figsize=(10, 4))
librosa.display.waveplot(signal, sr=sampling_rate)
plt.title(sound_file)
plt.show()

exit()

# Plot spectogram
plt.figure(figsize=(10, 4))
D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
plt.show()

# Plot mel-spectrogram
N_FFT = 1024
HOP_SIZE = 1024
N_MELS = 128
WIN_SIZE = 1024
WINDOW_TYPE = 'hann'
FEATURE = 'mel'
FMIN = 0

S = librosa.feature.melspectrogram(y=signal,sr=sampling_rate,
                                    n_fft=N_FFT,
                                    hop_length=HOP_SIZE,
                                    n_mels=N_MELS,
                                    htk=True,
                                    fmin=FMIN,
                                    fmax=sampling_rate/2)

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S**2,ref=np.max), fmin=FMIN,y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-scaled spectrogram')
plt.show(cmap ='gray')

# Plot mel-spectrogram with high-pass filter
N_FFT = 1024
HOP_SIZE = 1024
N_MELS = 128
WIN_SIZE = 1024
WINDOW_TYPE = 'hann'
FEATURE = 'mel'
FMIN = 1400

S = librosa.feature.melspectrogram(y=signal,sr=sr,
                                    n_fft=N_FFT,
                                    hop_length=HOP_SIZE,
                                    n_mels=N_MELS,
                                    htk=True,
                                    fmin=FMIN,
                                    fmax=sr/2)

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S**2,ref=np.max), fmin=FMIN,y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-scaled spectrogram with high-pass filter - 10 seconds')
plt.show()