#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: ProfileImgNet.py
@time: 2020/12/3 13:57
@desc:
'''
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input
from util import getBasePath
import keras

class ImgLoader():
    def __init__(self,imgNameList,tasktype='train'):
        self.imgNameList = imgNameList
        if tasktype not in ['train','test']:
            print("please set tasktype as train or test")
        self.tasktype = tasktype
        self.imgList = []
        self.imgBasePath = "%s/%s_profile_images/profile_images_%s"%(getBasePath(),self.tasktype,self.tasktype)

    def loadImgs(self):
        for aimg in self.imgNameList:
            img = image.load_img("%s/%s"%(self.imgBasePath,aimg), target_size=(32,32))
            self.imgList.append(img)

        return self.imgList/255


class ProfileCNN():
    def __init__(self):
        self.model = None
        self.buildNet()

    def buildNet(self,xshape):
        # create the base pre-trained model
        input_tensor = Input(shape=(32, 32, 3),name='img_input') # the shape of profile image
        self.base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

        # add a global spatial average pooling layer
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(512, activation='relu',name='fc1')(x)
        x = Dense(256, activation='relu', name='fc2')(x)
        # and a logistic layer -- let's say we have 3 classes
        predictions = Dense(3, activation='softmax')(x)

        # this is the model we will train
        self.model = Model(inputs=self.base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in self.base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    def fit(self,X,y):
        # train the model on the new data for a few epochs
        self.model.fit(X,y)

        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.

        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        for i, layer in enumerate(self.base_model.layers):
            print(i, layer.name)

        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        for layer in self.model.layers[:249]:
            layer.trainable = False
        for layer in self.model.layers[249:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate

        self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        self.model.fit(X,y)

    def extracFeas(self,X,model):
        return model(X)

    def savemodel(self):
        self.model.save('%s/savedModel/cnn-pfImg.h5' % getBasePath())

    def loadModel(self):
        return keras.models.load_model('%s/savedModel/cnn-pfImg.h5' % getBasePath())

if __name__=='__main__':
    import pandas as pd
    import numpy as np
    df = pd.read_csv("%s/train.csv"%getBasePath())
    indexlist_ = ['id', 'uname', 'url', 'covImgStatus', 'verifStatus', 'textColor', 'pageColor', 'themeColor',
                  'isViewSizeCustom', 'utcOffset', 'location', 'isLocVisible', 'uLanguage', 'creatTimestamp',
                  'uTimeZone', 'numFollowers', 'numPeopleFollowing', 'numStatUpdate', 'numDMessage', 'category',
                  'avgvisitPerSecond', 'avgClick', 'profileImg', 'numPLikes']
    df.columns = indexlist_

    imgNamelist = df['profileImg']
    bins = [0, 10000, 20000, np.max(df['pNumLikes'])]
    y = pd.cut(df['pNumLikes'], bins)
    loader = ImgLoader(imgNameList=imgNamelist,tasktype='train')