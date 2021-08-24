# 将训练集中的wav文件 以10s 为单位进行子wav文件的切分
from pydub import AudioSegment
import os
from tqdm import tqdm

split_duration = 30000

filePath = 'B099_0393.wav'
# 操作函数
def get_wav_make(wav_path):
    sound = AudioSegment.from_wav(wav_path)
    duration = sound.duration_seconds * 1000  # 音频时长（ms）
    begin = 0
    end = int(duration / 2)
    cut_wav = sound[begin:end]  # 以毫秒为单位截取[begin, end]区间的音频
    cut_wav.export(filePath + 'test.wav', format='wav')  # 存储新的wav文件
# get_wav_make(filePath)

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

# 训练集转换
train_data_path = 'train_data/train_data'
train_voice_path_dict = {}
convert_flord_to_pathdict(train_data_path, train_voice_path_dict)
# print(len(list(train_voice_path_dict.items())))
# print(train_voice_path_dict)

# 验证集转换
dev_data_path = 'dev_data'
dev_voice_path_dict = {}
convert_flord_to_pathdict(dev_data_path, dev_voice_path_dict)
# print(dev_voice_path_dict)

# 测试集转换
test_data_path = 'test_data/test_data_blind_name'
test_voice_path_dict = {}
for wav_path in os.listdir(test_data_path):
    test_voice_path_dict[os.path.join(test_data_path, wav_path)] = [wav_path, wav_path]
# print(test_voice_path_dict)

# 将wav文件拆分为多个10s的子文件，如果长度不足10s，则不动，如果剩余不足10s，则从后给前取10s
def split_wav_file(wav_path, label_, base_path):
    cat = label_[0]
    name = label_[1]
    new_cat_path = os.path.join(base_path, cat)
    if not os.path.exists(new_cat_path):
        os.makedirs(new_cat_path)
    sound = AudioSegment.from_wav(wav_path)
    duration = sound.duration_seconds * 1000  # 音频时长（ms）
    # 时长大于10s开始分割
    if duration > split_duration:
        begin = 0
        for index in range(200):
            end = begin + split_duration
            # 如果当前还没有到wav文件末尾
            if end < duration:
                new_file_path = os.path.join(new_cat_path, '{}_{}'.format(index+1, name))
                cut_wav = sound[begin: end]  # 以毫秒为单位截取[begin, end]区间的音频
                cut_wav.export(new_file_path, format='wav')  # 存储新的wav文件
                print(f'保存 {name} 的 第 {index+1} / {int(duration/split_duration)} 个子文件 {new_file_path} 成功！')
                begin += split_duration
            # 如果已经到文件末尾了，那就从后給前取10s
            else:
                new_file_path = os.path.join(new_cat_path, '{}_{}'.format(index + 1, name))
                cut_wav = sound[-split_duration:]
                cut_wav.export(new_file_path, format='wav')
                print(f'保存 {name} 的 第 {1} 个子文件 {new_file_path} 成功！')
                # 能走到这里说明当前文件已经取完了，所以跳出
                break
    # 时长不足10s，直接重新保存
    else:
        new_file_path = os.path.join(new_cat_path, '{}_{}'.format(1, name))
        sound.export(new_file_path, format='wav')

# 定义分割主函数
def run_split(path_dict, base_path):
    for wav_path in tqdm(path_dict):
        lable_ = path_dict[wav_path]
        split_wav_file(wav_path, lable_, base_path)

if __name__ == '__main__':
    train_base_path = 'train_split_data_30s'
    dev_base_path = 'dev_split_data_30s'
    run_split(train_voice_path_dict, train_base_path)
    run_split(dev_voice_path_dict, dev_base_path)
    test_base_path = 'test_split_data_30s'
    run_split(test_voice_path_dict, test_base_path)