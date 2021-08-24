# 暂时决定使用通过高频率波的梅尔频谱图作为wav预处理方式---暂时只拿前10s的频谱图
# 遍历所有训练集和验证集，测试集文件夹，将其都转换为梅尔频谱图的图片形式，以便后续tf读取
'''
文件格式如下：
train_img/
    -B000/
        -B000_0012.png
        -...
    -B001/
        -B001_0011.png
        -...
dev_img/
    -B000/
        -B000_0012.png
        -...
    -B001/
        -B001_0011.png
        -...
test_img/
    -暂时不着急转测试集
'''
import os
import librosa
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import librosa.display
import noisereduce as no
import queue
import threading
import tqdm
from PIL import Image

SAMPLE_RATE = 32000

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
    signal, sampling_rate = librosa.load(path=wav_path, duration=duration)  # 默认 sr=22050
    sr = sampling_rate
    # signal = no.reduce_noise(signal, signal, verbose=False)
    # Plot mel-spectrogram with high-pass filter
    N_FFT = 1024
    HOP_SIZE = 1024
    N_MELS = 128
    WIN_SIZE = 1024
    WINDOW_TYPE = 'hann'
    FEATURE = 'mel'
    FMIN = 20

    S = librosa.feature.melspectrogram(y=signal, sr=sr,
                                       n_fft=N_FFT,
                                       hop_length=HOP_SIZE,
                                       n_mels=N_MELS,
                                       htk=True,
                                       fmin=FMIN,
                                       fmax=sr / 2)

    plt.figure(figsize=(8, 8))
    # librosa.display.specshow(librosa.power_to_db(S ** 2, ref=np.max), fmin=FMIN, y_axis='linear')
    librosa.display.specshow(librosa.power_to_db(S ** 2, ref=np.max), fmin=FMIN)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel-scaled spectrogram with high-pass filter - 10 seconds')
    # plt.show()
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(dst_path)
    plt.close()

    print(dst_path, '写入成功！')

# 从dict中挨个解析地址，生成新的cat文件夹，cat文件夹中存放转换后的png 高频梅尔频谱图
def convert_pathdict_to_mlpng(path_dict):
    for wav_path in tqdm.tqdm(path_dict):
        label = path_dict[wav_path][0]
        # 决定存放图片的大文件夹地址
        if 'train' in wav_path:
            dst_path = 'train_split_img_old_{}s'.format(duration)
        elif 'test' in wav_path:
            dst_path = 'test_split_img_old_{}s'.format(duration)
        else:
            dst_path = 'dev_split_img_old_{}s'.format(duration)
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

# 配合多线程使用
def convert_tuple_to_mlpng(wav_path, label_):
    label = label_[0]
    # 决定存放图片的大文件夹地址
    if 'train' in wav_path:
        dst_path = 'train_img'
    else:
        dst_path = 'dev_img'
    cat_path = os.path.join(dst_path, label)
    # 判断是否生成了 cat文件夹,如果没有则手动生成
    if not os.path.exists(cat_path):
        os.mkdir(cat_path)
    # 开始给cat文件夹下写入当前 转换后图片文件
    # 转换后的图片名称为：
    full_dst_path = os.path.join(cat_path, label_[1].replace('wav', 'png'))
    # 如果当前文件不存在，才执行写入
    if not os.path.exists(full_dst_path):
        convert_wav_mlpng(wav_path=wav_path, dst_path=full_dst_path)
    else:
        print(full_dst_path, '已经存在，跳过。\n')

# 定义是否结束子线程的标志
exitFlag = 0

# 使用多线程完成数据读写
class Mythread(threading.Thread):
    # 配置每个 子进程 必要 操作参数
    def __init__(self, thread_id, que):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        # 只有一个实例化的 队列 传进来，每个线程都共享这个 队列
        self.que = que

    # 配置每个子进程的 任务（重复性任务）
    def run(self) -> None:
        print("开启线程：", self.thread_id)
        process_data(self.que, self.thread_id)
        print("退出线程：", self.thread_id)


# 定义各个线程如何从queue中取数据并处理数据
def process_data(que, id):
    while not exitFlag:
        queueLock.acquire()
        # 加锁从队列中获取一个数据给当前线程处理，取完数据立刻解锁
        if not workQueue.empty():
            data = que.get()
            queueLock.release()
            print(data)
            convert_tuple_to_mlpng(data[0], data[1])
        else:
            queueLock.release()

queueLock = threading.Lock()
workQueue = queue.Queue(12522)
threads = []
threadNum = 10

if __name__ == '__main__':
    for duration in [5, 10, 15, 20, 30]:
        # duration = 5
        # 训练集转换
        train_data_path = 'train_split_data_{}s'.format(duration)
        train_voice_path_dict = {}
        convert_flord_to_pathdict(train_data_path, train_voice_path_dict)
        # print(len(list(train_voice_path_dict.items())))
        # print(train_voice_path_dict)
        convert_pathdict_to_mlpng(train_voice_path_dict)

        # 验证集转换
        dev_data_path = 'dev_split_data_{}s'.format(duration)
        dev_voice_path_dict = {}
        convert_flord_to_pathdict(dev_data_path, dev_voice_path_dict)
        # print(dev_voice_path_dict)
        convert_pathdict_to_mlpng(dev_voice_path_dict)

        # 测试集转换---将test_split_data 中的wav都转为img
        test_data_path = 'test_split_data_{}s'.format(duration)
        test_voice_path_dict = {}
        convert_flord_to_pathdict(test_data_path, test_voice_path_dict)
        # print(test_voice_path_dict)
        convert_pathdict_to_mlpng(test_voice_path_dict)

    # 无法使用多线程！
    # train_voice_path_dict.update(dev_voice_path_dict)
    # print(len(list(train_voice_path_dict.items())))
    #
    # # 填充队列
    # queueLock.acquire()
    # for wav_path in train_voice_path_dict:
    #     workQueue.put((wav_path, train_voice_path_dict[wav_path]))
    # queueLock.release()
    #
    # # 创建新线程
    # for index in range(threadNum):
    #     thread = Mythread(index+1, workQueue)
    #     thread.start()
    #     threads.append(thread)
    #
    # # 等待队列清空，实际上主线程会执行到这里，然后卡在这个循环，子线程们不断执行上面的 process_data 函数，消耗queue里面的数据，直到耗完
    # while not workQueue.empty():
    #     pass
    #
    # # 当子线程消耗完了数据，修改flag，通知子线程是时候退出---子线程不会卡在 process_data 的循环中了，跳出后执行到 run函数中的下一句了
    # exitFlag = 1
    #
    # # 等待所有线程执行完当前最后在处理的数据，即执行完了run的最后一行---线程没任务了，但是还没有清除掉。  使用 join中止（清除）所有子线程
    # for t in threads:
    #     t.join()
    #
    # print('所有数据处理完毕，queue为空，主线程退出！')