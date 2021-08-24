import librosa
import numpy as np
import pandas as pd

def filter_wav_find_bird_sound(wav_path):
    signal, sampling_rate = librosa.load(path=wav_path, duration=SIGNAL_LENGTH, sr=SAMPLE_RATE)  # 默认 sr=22050
    hop_length = int(SIGNAL_LENGTH * SAMPLE_RATE / (SPEC_SHAPE[1] - 1))
    mel_spec = librosa.feature.melspectrogram(y=signal,
                                              sr=SAMPLE_RATE,
                                              n_fft=2048,
                                              hop_length=hop_length,
                                              n_mels=SPEC_SHAPE[0],
                                              fmin=FMIN,
                                              fmax=FMAX)
    df_mel_spec = pd.DataFrame(mel_spec)
    print(df_mel_spec.max())

    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    # Normalize
    mel_spec -= mel_spec.min()
    mel_spec /= mel_spec.max()
    # print(mel_spec.shape)

    # print(mel_spec)

if __name__ == '__main__':
    sound_file = 'B000_0017.wav'  # 1mins
    SIGNAL_LENGTH = 10
    SAMPLE_RATE = 32000
    SPEC_SHAPE = (224, 224)
    FMIN = 0
    FMAX = 16000
    filter_wav_find_bird_sound(sound_file)