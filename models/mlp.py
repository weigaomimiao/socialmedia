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
from util import getBasePath,trainvalLossPlot
from keras.optimizers import RMSprop
from sklearn.metrics import mean_squared_error

class MLP():
    def __init__(self,xshape,learning_rate=1e2,epochs=200,batch_size=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
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
        opt = keras.optimizers.SGD(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss='mse', metrics=['mse'])

        return model

    def fit(self,X,y):
        history = self.model.fit(X, y,epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        # print(10*'=')
        # print('train loss:', history.history['loss'])
        # print('val loss:',history.history['mse'])
        # print('mean val loss',np.mean(history.history['mse']))

        # trainvalLossPlot(history.history['loss'],history.history['mse'],'mlp')

    def predict(self,X):
        return self.model.predict(X)

    def savemodel(self):
        self.model.save('%s/savedModel/model-mlp.h5'%getBasePath())

    def loadModel(self):
        return keras.models.load_model('%s/savedModel/model-mlp.h5'%getBasePath())

    def get_params(self, deep=True):
        return {"learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size}

    def set_params(self, **params):
        if not params:
            #  Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            setattr(self, key, value)
            valid_params[key] = value

        return self

        # for parameter, value in params.items():
        #     setattr(self, parameter, value)
        # return self
    def score(self,X,y):
        return mean_squared_error(y,self.predict(X))
