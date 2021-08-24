import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping
import efficientnet.tfkeras as efn
from efficientnet.keras import preprocess_input
from sklearn.utils import class_weight
import numpy as np
import os

IM_SIZE = [224, 224, 3]
BATCH_SIZE = 32
CLASS_NUM = 100

# 准备数据
# 训练集
# train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255., width_shift_range=0.2,
#                                    height_shift_range=0.2, shear_range=0.2, zoom_range=0.1, fill_mode='nearest')
# 取消部分不必要的预处理
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255.,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(directory='train_split_img_new_5s', target_size=IM_SIZE[:2],
                                                    class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)

# 验证集
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255.)
valid_generator = valid_datagen.flow_from_directory(directory='dev_split_img_new_5s', target_size=IM_SIZE[:2],
                                                    class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)

# 使样本不均衡的数据集中每个类别的样本被抽到的概率一样大,classes类别标签，y-训练集label
# Estimate class weights for unbalanced dataset
class_weights = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes),
                                                  train_generator.classes).tolist()
# 需要转成字典
class_weight_dict = dict(zip([x for x in np.unique(train_generator.classes)], class_weights))

#Define CNN's architecture
backbone_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=IM_SIZE, classes=CLASS_NUM)
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

# compile
model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])


# define callback
check_point_path = 'models/InceptionResNetV2_no_pretrained_new_img_5s/EfficientNetB3.ckpt'
if os.path.exists(check_point_path + '.index'):
    print('-------------load the mode1-------------')
    model.load_weights(check_point_path)
cp_checkpoint = ModelCheckpoint(filepath=check_point_path, monitor='val_loss', save_best_only=True,
                                save_weights_only=True)
# ReduceLROnPlateau 自动随着训练进程缩小学习率以优化模型训练效率---常配合earlystop使用
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
# early_stop = EarlyStopping(monitor='val_loss', patience=15, mode='auto')

tensorboard_log_path = 'logs/InceptionResNetV2_no_pretrained_new_img_5s/'
tb_checkpoint = TensorBoard(log_dir=tensorboard_log_path, update_freq='epoch', histogram_freq=1)

# train model
epochs = 100
# model.fit(x=train_generator, epochs=epochs, callbacks=[cp_checkpoint, reduce_lr, tb_checkpoint, early_stop],
#           validation_data=valid_generator, shuffle=True, class_weight=class_weight_dict)
model.fit(x=train_generator, epochs=epochs, callbacks=[cp_checkpoint, reduce_lr, tb_checkpoint],
          validation_data=valid_generator, shuffle=True, class_weight=class_weight_dict)