# use EfficientNetB3 model to predict test_data
# 主要验证我将一个wav文件分割成多个wav文件去预测，然后取频率最高的lable作为最后的label 这个想法是否正确
# 如果预测想法正确，大概率拆分wav文件然后去训练也是正确的
# 可以进一步考虑把多个音频的 max结果概率也输出，然后最终预测结果是多个结果中 概率最高的哪个，而不是投票出现最多的cat

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation, LSTM, Bidirectional
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import efficientnet.tfkeras as efn
from efficientnet.keras import preprocess_input
import os
import keras.backend as K
import pandas as pd
import numpy as np

# Define CNN's architecture
IM_SIZE = (224, 224, 3)
BATCH_SIZE = 32
CLASS_NUM = 100

#Define CNN's architecture
backbone_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights=None, input_shape=IM_SIZE, classes=CLASS_NUM)
# backbone_model.summary()
for layer in backbone_model.layers:
    layer.trainable = True

model = Sequential()
model.add(backbone_model)
# model.add(GlobalAveragePooling2D())  # 可以一定程度代替全连接层，将 高维特征降为一维
model.add(Flatten())
model.add(Dense(512, use_bias=True))
model.add(Dense(256, use_bias=True))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(units=CLASS_NUM, activation='softmax', name='softmax'))
# model.summary()

# reload callback
# define callback
check_point_path = 'models/InceptionResNetV2_no_pretrained_new_img_5s/EfficientNetB3.ckpt'
if os.path.exists(check_point_path + '.index'):
    print('-------------load the mode1-------------')
    model.load_weights(check_point_path)

# 生成 测试数据生成器
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255.)
test_generator = test_datagen.flow_from_directory(directory='test_split_img_new_5s', target_size=IM_SIZE[:2],
                                                  batch_size=128, class_mode='categorical', shuffle=False)
# 预测的wav文件名称
filenames = test_generator.filenames
# print(filenames)
# filenames = [filename.replace('test_data/', '').replace('png', 'wav') for filename in filenames]
# print(filenames)

# 预测结果
# pred = model.predict_generator(test_generator, verbose=1)
pred = model.predict(test_generator, verbose=1)
# 拿到每一行预测结果的最大概率值
probility = np.amax(pred, axis=1)
# 拿到每一行预测概率最大结果的label
pred = pred.argmax(axis=1)

class_indices = {'B000': 0, 'B001': 1, 'B002': 2, 'B003': 3, 'B004': 4, 'B005': 5, 'B006': 6, 'B007': 7, 'B008': 8,
                 'B009': 9, 'B010': 10, 'B011': 11, 'B012': 12, 'B013': 13, 'B014': 14, 'B015': 15, 'B016': 16,
                 'B017': 17, 'B018': 18, 'B019': 19, 'B020': 20, 'B021': 21, 'B022': 22, 'B023': 23, 'B024': 24,
                 'B025': 25, 'B026': 26, 'B027': 27, 'B028': 28, 'B029': 29, 'B030': 30, 'B031': 31, 'B032': 32,
                 'B033': 33, 'B034': 34, 'B035': 35, 'B036': 36, 'B037': 37, 'B038': 38, 'B039': 39, 'B040': 40,
                 'B041': 41, 'B042': 42, 'B043': 43, 'B044': 44, 'B045': 45, 'B046': 46, 'B047': 47, 'B048': 48,
                 'B049': 49, 'B050': 50, 'B051': 51, 'B052': 52, 'B053': 53, 'B054': 54, 'B055': 55, 'B056': 56,
                 'B057': 57, 'B058': 58, 'B059': 59, 'B060': 60, 'B061': 61, 'B062': 62, 'B063': 63, 'B064': 64,
                 'B065': 65, 'B066': 66, 'B067': 67, 'B068': 68, 'B069': 69, 'B070': 70, 'B071': 71, 'B072': 72,
                 'B073': 73, 'B074': 74, 'B075': 75, 'B076': 76, 'B077': 77, 'B078': 78, 'B079': 79, 'B080': 80,
                 'B081': 81, 'B082': 82, 'B083': 83, 'B084': 84, 'B085': 85, 'B086': 86, 'B087': 87, 'B088': 88,
                 'B089': 89, 'B090': 90, 'B091': 91, 'B092': 92, 'B093': 93, 'B094': 94, 'B095': 95, 'B096': 96,
                 'B097': 97, 'B098': 98, 'B099': 99}
pred = list(pred)
probility = list(probility)
account_dict = {}
probility_dict = {}
for p, pro, file_name in zip(pred, probility, filenames):
    test_wav_name = file_name[:12]
    # 获取已经投票的数量
    temp_vote_list = account_dict.get(test_wav_name)
    temp_prob_list = probility_dict.get(test_wav_name)
    # 如果为空，则给字典中创建该元素
    if not temp_vote_list:
        account_dict[test_wav_name] = [list(class_indices.keys())[p]]
        probility_dict[test_wav_name] = [pro]
    else:
        temp_vote_list.append(list(class_indices.keys())[p])
        temp_prob_list.append(pro)
# 统计完每张图片的数量以后，从value中选出出现频率最高的元素
answer_list = []
answer_list_prob = []
for test_wav_name in account_dict:
    account_list = account_dict[test_wav_name]
    maxlabel = max(account_list, key=account_list.count)
    answer_list.append(test_wav_name+' '+maxlabel)
    # 开始找预测概率最大的prob
    probility_list = probility_dict[test_wav_name]
    max_index = probility_list.index(max(probility_list, key=abs))
    max_pro_label = account_list[max_index]
    answer_list_prob.append(test_wav_name+' '+max_pro_label)

df = pd.Series(answer_list)
df.to_csv('submit_resnet50_maxvote.csv', index=False, header=False)

df = pd.Series(answer_list_prob)
df.to_csv('submit_resnet50_maxprob.csv', index=False, header=False)

K.clear_session()