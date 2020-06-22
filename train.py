!nvidia-smi

!unzip cats_and_dogs.zip

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import datetime
import sys
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight as cw

from keras import Sequential

from keras.models import Model

from keras.layers import LSTM,Activation,Dense,Dropout,Input,Embedding,BatchNormalization,Add,concatenate,Flatten
from keras.layers import Conv1D,Conv2D,Convolution1D,MaxPool1D,SeparableConv1D,SpatialDropout1D,GlobalAvgPool1D,GlobalMaxPool1D,GlobalMaxPooling1D
from keras.layers.pooling import _GlobalPooling1D
from keras.layers import MaxPooling2D,GlobalMaxPooling2D,GlobalAveragePooling2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau


from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

img_width,img_height = 299, 299    #修正 InceptionV3 的尺寸参数
epochs = 5
batch_size = 32
fc_size = 1024

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))       # glob模块是用来查找匹配文件的，后面接匹配规则。
    return cnt
    
# 定义增加最后一个全连接层的函数
def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet

    Args:
        base_model: keras model excluding top
        nb_classes: # of classes

    Returns:
        new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(fc_size, activation='relu')(x) #new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# 定义微调函数
def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.

        note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

    Args:
        model: keras model
    """
    # for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
    #     layer.trainable = False
    for layer in model.layers[:]:
        layer.trainable = True
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

train_dir= "./train"
valid_dir = "./validation"
nb_classes = len(glob.glob(train_dir + "/*"))
# nb_val_samples = get_nb_files(val_dir)
epochs = int(epochs)
batch_size = int(batch_size)

train_data = ImageDataGenerator(
    # 浮点数，剪切强度（逆时针方向的剪切变换角度）
    shear_range=0.1,
    # 随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
    zoom_range=0.1,
    # 浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
    width_shift_range=0.1,
    # 浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
    height_shift_range=0.1,
    # 布尔值，进行随机水平翻转
    horizontal_flip=True,
    # 布尔值，进行随机竖直翻转
    vertical_flip=True,
    # 在 0 和 1 之间浮动。用作验证集的训练数据的比例
#         validation_split=0.3
    )

# 接下来生成验证集，可以参考训练集的写法
validation_data = ImageDataGenerator(
    # 浮点数，剪切强度（逆时针方向的剪切变换角度）
    shear_range=0.1,
    # 随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
    zoom_range=0.1,
    # 浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
    width_shift_range=0.1,
    # 浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
    height_shift_range=0.1,
    # 布尔值，进行随机水平翻转
    horizontal_flip=True,
    # 布尔值，进行随机竖直翻转
    vertical_flip=True,
)

train_generator = train_data.flow_from_directory(
    # 提供的路径下面需要有子目录
    train_dir,
    # 整数元组 (height, width)，默认：(256, 256)。 所有的图像将被调整到的尺寸。
    target_size=(299,299),
    # 一批数据的大小
    batch_size=batch_size,
    # "categorical", "binary", "sparse", "input" 或 None 之一。
    # 默认："categorical",返回one-hot 编码标签。
    class_mode='categorical',
    seed=0)

validation_generator = train_data.flow_from_directory(
    valid_dir,
    target_size=(299,299),
    batch_size=batch_size,
    class_mode='categorical',
    seed=0)

# 准备跑起来，首先给 base_model 和 model 赋值，迁移学习和微调都是使用 InceptionV3 的 notop 模型（看 inception_v3.py 源码，此模型是打开了最后一个全连接层），利用 add_new_last_layer 函数增加最后一个全连接层。

base_model = InceptionV3(weights="imagenet", include_top=False)  # include_top=False excludes final FC layer
model = add_new_last_layer(base_model, nb_classes)


print("开始微调:\n")

# fine-tuning
setup_to_finetune(model)

model.summary()

print("Setting Callbacks")

checkpoint = ModelCheckpoint("model.h5",
                                                     monitor="val_loss",
                                                     save_best_only=True,
                                                     mode="min")

early_stopping = EarlyStopping(monitor="val_loss",
                                                     patience=3,
                                                     verbose=1,
                                                     restore_best_weights=True,
                                                     mode="min")

reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                                                      factor=0.6,
                                                      patience=2,
                                                      verbose=1,
                                                      mode="min")

callbacks=[checkpoint,early_stopping,reduce_lr]

history_ft = model.fit_generator(
    train_generator,
    steps_per_epoch=1800   // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=600    // batch_size,
    callbacks=callbacks)

# 画曲线
def plot_performance(history=None,figure_directory=None,ylim_pad=[0,0]):
    xlabel="Epoch"
    legends=["Training","Validation"]
    
    plt.figure(figsize=(20,5))
    
    y1=history.history["accuracy"]
    y2=history.history["val_accuracy"]
    
    min_y=min(min(y1),min(y2))-ylim_pad[0]
    max_y=max(max(y1),max(y2))+ylim_pad[0]
    
    plt.subplot(121)
    
    plt.plot(y1)
    plt.plot(y2)
    
    plt.title("Model Accuracy\n",fontsize=17)
    plt.xlabel(xlabel,fontsize=15)
    plt.ylabel("Accuracy",fontsize=15)
    plt.ylim(min_y,max_y)
    plt.legend(legends,loc="upper left")
    plt.grid()
    
    y1=history.history["loss"]
    y2=history.history["val_loss"]
    
    min_y=min(min(y1),min(y2))-ylim_pad[1]
    max_y=max(max(y1),max(y2))+ylim_pad[1]
    
    plt.subplot(122)
    
    plt.plot(y1)
    plt.plot(y2)
    
    plt.title("Model Loss:\n",fontsize=17)
    plt.xlabel(xlabel,fontsize=15)
    plt.ylabel("Loss",fontsize=15)
    plt.ylim(min_y,max_y)
    plt.legend(legends,loc="upper left")
    plt.grid()
    plt.show()

plot_performance(history_ft)














