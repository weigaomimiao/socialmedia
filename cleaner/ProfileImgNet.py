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
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input
from util import getBasePath
from keras import backend as K
import tensorflow as tf
import os
import numpy as np

class ImgLoader():
    def __init__(self, imgNameList, tasktype='train',path="%s/../data/%s_profile_images/profile_images_%s"):
        self.imgNameList = imgNameList
        if tasktype not in ['train', 'test']:
            print("please set tasktype as train or test")
        self.tasktype = tasktype
        self.imgList = []
        self.imgBasePath = path % (getBasePath(), self.tasktype, self.tasktype)
        self.imgExisIndex = []

    def loadImgs(self):
        for aimg in self.imgNameList:
            apath = (self.imgBasePath+"/%s")% aimg
            if(not os.path.exists(apath)):
                self.imgExisIndex.append(False)
                continue
            else:
                self.imgExisIndex.append(True)
                img = image.load_img(apath, target_size=(32, 32),color_mode='rgb',interpolation='nearest')
                img = np.array(image.img_to_array(img))
                self.imgList.append(img)

        return np.array(self.imgList)/ 255, self.imgExisIndex


class ProfileCNN():
    def __init__(self,xshape,yclassNum=10,epochs=1,feaDim=128,isRelativePath=True):
        self.model = None
        self.xshape_ = xshape
        self.yclassNum_ = yclassNum
        self.epochs = epochs
        self.feaDim = feaDim
        if isRelativePath: # Indicate whether to use  or %s/savedModel/cnn-pfImg.h5
            self.modelpath = '%s/../savedModel/cnn-pfImg.h5'
        else:
            self.modelpath = '%s/savedModel/cnn-pfImg.h5'
        self.buildNet()

    def buildNet(self):
        # create the base pre-trained model
        input_tensor = Input(shape=self.xshape_, name='img_input')  # the shape of profile image
        self.base_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

        # add a global spatial average pooling layer
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(self.feaDim, activation='relu', name='fc1')(x)
        x = Dense(self.feaDim/2, activation='relu', name='fc2')(x)
        # and a logistic layer -- let's say we have 3 classes
        predictions = Dense(self.yclassNum_, activation='softmax')(x)

        # this is the model we will train
        self.model = Model(inputs=self.base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in self.base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    def fit(self, X, y):
        # train the model on the new data for a few epochs
        self.model.fit(X, y)

        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.

        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        for i, layer in enumerate(self.base_model.layers):
            print(i, layer.name)

        # we chose to train the top 2 blocks, i.e. we will freeze
        # the first 16 layers and unfreeze the rest:
        for layer in self.model.layers[:16]:
            layer.trainable = False
        for layer in self.model.layers[16:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate

        self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        self.model.fit(X, y,epochs=self.epochs)

    def _get_layer_output(self, x, index=-1):
        """
        get the computing result output of any layer you want, default the last layer.
        :param model: primary model
        :param x: input of primary model( x of model.predict([x])[0])
        :param index: index of target layer, i.e., layer[23]
        :return: result
        """
        layer = K.function([self.model.input], [self.model.layers[index].output])
        return layer([x])[0]

    def extracFeas(self, X):
        features = self._get_layer_output(X, -3)  # the last third layer
        return features

    def savemodel(self):
        self.model.save(self.modelpath % getBasePath())

    def loadModel(self):
        bs = getBasePath()
        return tf.keras.models.load_model(self.modelpath % getBasePath())


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    df_train = pd.read_csv("%s/../data/train.csv" % getBasePath())
    df_test = pd.read_csv("%s/../data/test.csv" % getBasePath())
    indexlist_ = ['id', 'uname', 'url', 'covImgStatus', 'verifStatus', 'textColor', 'pageColor', 'themeColor',
                  'isViewSizeCustom', 'utcOffset', 'location', 'isLocVisible', 'uLanguage', 'creatTimestamp',
                  'uTimeZone', 'numFollowers', 'numPeopleFollowing', 'numStatUpdate', 'numDMessage', 'category',
                  'avgvisitPerSecond', 'avgClick', 'profileImg', 'numPLikes']
    df_train.columns = indexlist_
    df_test.columns = indexlist_[:-1]

    imgNamelist_train = df_train['profileImg'].values
    imgNamelist_test = df_test['profileImg'].values

    yclassNum = 10
    df_train['numPLikeslog'] = df_train['numPLikes'].map(lambda x: np.log10(1.5+x))
    y = pd.get_dummies(pd.qcut(df_train['numPLikeslog'], yclassNum,duplicates='drop')).values # note ycalssNum
    # y = pd.get_dummies(pd.cut(df_train['numPLikeslog'], yclassNum)).values
    loader = ImgLoader(imgNameList=imgNamelist_train,tasktype='train')
    trainX,imgExisIndex = loader.loadImgs()
    # print(np.array(imgExisIndex).sum())
    xshape = trainX[0].shape
    trainy = y[imgExisIndex]
    trainy = y

    net = ProfileCNN(xshape,yclassNum=yclassNum,feaDim=128,isRelativePath=True)
    net.fit(trainX,trainy)
    net.savemodel()

    extractor = ProfileCNN(xshape=(32,32,3),yclassNum=yclassNum,feaDim=128)
    extractor.loadModel()
    feas = extractor.extracFeas(trainX[:3,:,:,:]) # test the 1st three imgs
    print(feas.shape)
    # the output dimension of features is 512, you can decrease it by feaDim.
    # But keep feaDim at least 4 times of yclassNum
