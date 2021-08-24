# 参考https://blog.csdn.net/hhhhhhhhhhwwwwwwwwww/article/details/116031884的方法进行频谱图片的生成
import os
import librosa
import numpy as np
from PIL import Image
import tqdm

# 将当前文件夹解析为配套的dict地址-cat 键值对
def convert_flord_to_pathdict(train_data_path, train_voice_path_dict):
    cat_list = os.listdir(train_data_path)
    # 一共有 100 个类别文件夹
    # print(cat_list)
    for cat_dir in cat_list:
        cat_fload_path = os.path.join(train_data_path, cat_dir)
        sound_name_list = os.listdir(cat_fload_path)
        sound_full_path_list = [(os.path.join(cat_fload_path, sound_name), sound_name) for sound_name in sound_name_list]
        for sound_full_path, sound_name in sound_full_path_list:
            train_voice_path_dict[sound_full_path] = [cat_dir, sound_name]

# 读取wav音频文件，并将其转换为高频滤波的ml频谱图
def convert_wav_mlpng(wav_path, dst_path):
    signal, sampling_rate = librosa.load(path=wav_path, duration=SIGNAL_LENGTH, sr=SAMPLE_RATE)  # 默认 sr=22050
    hop_length = int(SIGNAL_LENGTH * SAMPLE_RATE / (SPEC_SHAPE[1] - 1))
    mel_spec = librosa.feature.melspectrogram(y=signal,
                                              sr=SAMPLE_RATE,
                                              n_fft=2048,
                                              hop_length=hop_length,
                                              n_mels=SPEC_SHAPE[0],
                                              fmin=FMIN,
                                              fmax=FMAX)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    # Normalize
    mel_spec -= mel_spec.min()
    mel_spec /= mel_spec.max()
    im = Image.fromarray(mel_spec * 255.0).convert("L")
    im.save(dst_path)
    # print(dst_path, '写入成功！')

# 从dict中挨个解析地址，生成新的cat文件夹，cat文件夹中存放转换后的png 高频梅尔频谱图
def convert_pathdict_to_mlpng(path_dict):
    for wav_path in tqdm.tqdm(path_dict):
        label = path_dict[wav_path][0]
        # 决定存放图片的大文件夹地址
        if 'train' in wav_path:
            dst_path = 'train_split_img_new_{}_5s'.format(duration)
        elif 'test' in wav_path:
            dst_path = 'test_split_img_new_{}_5s'.format(duration)
        else:
            dst_path = 'dev_split_img_new_{}_5s'.format(duration)
        # 生成dstpath ---> train_split_img_old_5s
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        cat_path = os.path.join(dst_path, label)
        # 判断是否生成了 cat文件夹,如果没有则手动生成
        if not os.path.exists(cat_path):
            os.mkdir(cat_path)
        # 开始给cat文件夹下写入当前 转换后图片文件
        # 转换后的图片名称为：
        full_dst_path = os.path.join(cat_path, path_dict[wav_path][1].replace('wav', 'png'))
        # 如果当前文件不存在，才执行写入
        if not os.path.exists(full_dst_path):
            convert_wav_mlpng(wav_path=wav_path, dst_path=full_dst_path)
        else:
            print(full_dst_path, '已经存在，跳过。')


if __name__ == '__main__':
    SAMPLE_RATE = 32000
    SPEC_SHAPE = (224, 224)  # height x width
    FMIN = 40
    FMAX = 16000
    # for duration in [5, 10, 15, 20, 30]:
    for duration in [15]:
        SIGNAL_LENGTH = duration
        # 训练集转换
        train_data_path = 'train_split_data_{}_5s'.format(duration)
        train_voice_path_dict = {}
        convert_flord_to_pathdict(train_data_path, train_voice_path_dict)
        # print(len(list(train_voice_path_dict.items())))
        # print(train_voice_path_dict)
        convert_pathdict_to_mlpng(train_voice_path_dict)

        # 验证集转换
        dev_data_path = 'dev_split_data_{}_5s'.format(duration)
        dev_voice_path_dict = {}
        convert_flord_to_pathdict(dev_data_path, dev_voice_path_dict)
        # print(dev_voice_path_dict)
        convert_pathdict_to_mlpng(dev_voice_path_dict)

        # 测试集转换---将test_split_data 中的wav都转为img
        test_data_path = 'test_split_data_{}_5s'.format(duration)
        test_voice_path_dict = {}
        convert_flord_to_pathdict(test_data_path, test_voice_path_dict)
        # print(test_voice_path_dict)
        convert_pathdict_to_mlpng(test_voice_path_dict)