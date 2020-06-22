import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from keras import regularizers
from PIL import Image
from tensorflow.keras.preprocessing import image
import glob
# Image processing
from PIL import Image, ImageFile
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.layers import Activation, Dense
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adam, Adagrad, RMSprop
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from sklearn.metrics import accuracy_score

true=[]
pred=[]

garbage_types = ['cats','dogs']
labels = {0:'cats',1:'dogs'}

model_path = 'model.h5'
model = load_model(model_path)

for garbage in garbage_types:
    files = glob.glob("./test/" + str(garbage) + "/*.jpg")
    for myFile in files:
        t = list(labels.keys())[list(labels.values()).index(str(garbage))]
        true.append(t)
        img = image.load_img(myFile,target_size=(224,224))
        input_image = image.img_to_array(img)
        
    
        # 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
        # 如果你的模型是在 results 文件夹下的 dnn.h5 模型，则 model_path = 'results/dnn.h5'
        # -------------------------- 实现模型预测部分的代码 ---------------------------
        # expand_dims的作用是把img.shape转换成(1, img.shape[0], img.shape[1], img.shape[2])
        x = np.expand_dims(img, axis=0)
     # 模型预测
        y = model.predict(x)
        predict = labels[np.argmax(y)]
        pred.append(np.argmax(y))


acc = accuracy_score(true,pred)
print(acc)
con_matrix = confusion_matrix(true, pred,
                              labels=[0, 1])
plt.figure(figsize=(10, 10))
plt.title('Prediction of garbage types')
plt.ylabel('True label')
plt.xlabel('Predicted label')
# plt.show(sns.heatmap(con_matrix, annot=True, fmt="d",annot_kws={"size": 7},cmap='Blues',square=True))
ax = sns.heatmap(con_matrix, annot=True, fmt="d", annot_kws={"size": 7}, cmap='Blues', square=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()