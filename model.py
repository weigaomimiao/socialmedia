#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: model.py
@time: 2020/11/29 17:46
@desc: all kinds of models, the final version of model is also included.
'''
from sklearn.svm import SVR
from keras import models
from keras import layers
from util import getBasePath
import numpy as np
import keras
class Model():
    def __init__(self,xshape,modeltype='mlp',savemodel=True):
        self.model = self.initmodel(modeltype,xshape,savemodel)

    def initmodel(self,modeltype,xshape,savemodel):
        if modeltype=='SVR':
            return SVR()
        elif modeltype=='mlp':
            return MLP(xshape,savemodel)
        return None

    def fit(self,X,y):
        self.model.fit(X,y)

    def predict(self,X):
        return self.model.predict(X)

    def savemodel(self):
        self.model.savemodel()

    def loadModel(self):
        return self.model.loadModel()


class MLP():
    def __init__(self,xshape,savemodel):
        self.model = self.buildModel(xshape)
        self.isModelSave = savemodel

    def buildModel(self,xshape):
        model = models.Sequential()
        model.add(layers.Dense(32, activation='relu', input_shape=(xshape,)))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        ##opt = keras.optimizers.Adam(learning_rate=0.01)
        # optimizer='rmsprop'
        opt = keras.optimizers.rmsprop(learning_rate=0.001)
        model.compile(optimizer=opt, loss='mse', metrics=['mae'])

        return model

    def fit(self,X,y):
        self.model.fit(X, y,epochs=100, batch_size=20, verbose=0)
        # if self.isModelSave:
        #     self.savemodel()

    def predict(self,X):
        ypred = self.model.predict(X)
        return np.ceil(ypred)

    def savemodel(self):
        self.model.save('%s/model/model.h5'%getBasePath())

    def loadModel(self):
        return keras.models.load_model('%s/model/model.h5'%getBasePath())