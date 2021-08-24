# 分析数据集的文件结构及其组成

import os
import pandas as pd
import wave
import contextlib

train_data_path = 'train_data/train_data'
cat_list = os.listdir(train_data_path)
# 一共有 100 个类别文件夹
# print(cat_list)
train_voice_num_dict = {}
for cat_dir in cat_list:
    path = os.path.join(train_data_path, cat_dir)
    voice_num = len(os.listdir(path))
    train_voice_num_dict[cat_dir] = voice_num

# 判断一下训练集文件纯不纯
for cat_dir in cat_list:
    path = os.path.join(train_data_path, cat_dir)
    voice_name_list = os.listdir(path)
    for voice_name in voice_name_list:
        if '.wav' not in voice_name:
            print('find error file, named {}, in folder {}'.format(voice_name, cat_dir))

# 总共的音频数量是 11006, 数据不是很均衡，最少7个，最多350个
# print(train_voice_num_dict)
train_df = pd.Series(train_voice_num_dict)
# print(train_df.describe())
# print(train_df.sum())

# 每个音频文件的时长 不同， 最长的673s，最短的11s
dst_cat_path = os.path.join(train_data_path, cat_list[0])
wav_path_list = os.listdir(dst_cat_path)
wav_duration_dict = {}
for wav_path in wav_path_list:
    fname = os.path.join(dst_cat_path, wav_path)
    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        wav_duration_dict[fname] = duration
train_df = pd.Series(wav_duration_dict)
# print(train_df.describe())


dev_data_path = 'dev_data'
cat_list = os.listdir(dev_data_path)
# 一共有 100 个类别文件夹
# print(cat_list)
dev_voice_num_dict = {}
wav_duration_dict = {}
for cat_dir in cat_list:
    path = os.path.join(dev_data_path, cat_dir)
    voice_num = len(os.listdir(path))
    dev_voice_num_dict[cat_dir] = voice_num
    for wav_path in os.listdir(path):
        fname = os.path.join(path, wav_path)
        with contextlib.closing(wave.open(fname, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            wav_duration_dict[fname] = duration
train_df = pd.Series(wav_duration_dict)
print(train_df.describe())

# 总共的验证音频数量是 1516, 数据不是很均衡，最少0个 B017，最多50个,最少都有6s，大多数在30多s
# print(dev_voice_num_dict)
# dev_df = pd.Series(dev_voice_num_dict)
# print(dev_df.describe())
# print(dev_df.sum())

# 判断一下验证集文件纯不纯
for cat_dir in cat_list:
    path = os.path.join(dev_data_path, cat_dir)
    voice_name_list = os.listdir(path)
    for voice_name in voice_name_list:
        if '.wav' not in voice_name:
            print('find error file, named {}, in folder {}'.format(voice_name, cat_dir))


# 测试集文件时长分析,最短的是5s，平均也在30s左右，3084个样本
wav_duration_dict = {}
test_data_path = 'test_data/test_data_blind_name'
for wav_path in os.listdir(test_data_path):
    path = os.path.join(test_data_path, wav_path)
    with contextlib.closing(wave.open(path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        wav_duration_dict[path] = duration
train_df = pd.Series(wav_duration_dict)
print(train_df.describe())