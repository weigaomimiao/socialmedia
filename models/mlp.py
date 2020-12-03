#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: mlp.py
@time: 2020/12/3 13:47
@desc:
'''
from keras import models
from keras import layers
import numpy as np
import keras
import tensorflow as tf
from util import getBasePath
from keras.optimizers import RMSprop

class MLP():
    def __init__(self,xshape):
        self.model = self.buildModel(xshape)


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

        # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=1e-2,
        #     decay_steps=100,
        #     decay_rate=0.9)
        opt = keras.optimizers.SGD(learning_rate=1e-2)
        model.compile(optimizer=opt, loss='mse', metrics=['msle'])

        return model

    def fit(self,X,y):
        history = self.model.fit(X, y,epochs=100, batch_size=50, verbose=0)
        print(10*'=')
        print('train loss:', history.history['loss'])
        print('val loss:',history.history['msle'])

    def predict(self,X):
        return self.model.predict(X)

    def savemodel(self):
        self.model.save('%s/savedModel/model-mlp.h5'%getBasePath())

    def loadModel(self):
        return keras.models.load_model('%s/savedModel/model-mlp.h5'%getBasePath())
