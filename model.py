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
class Model():
    def __init__(self,xshape,modeltype='mlp'):
        self.model = self.initmodel(modeltype,xshape)

    def initmodel(self,modeltype,xshape):
        if modeltype=='SVR':
            return SVR()
        elif modeltype=='mlp':
            return MLP(xshape)
        return None

    def fit(self,X,y):
        self.model.fit(X,y)

    def predict(self,X):
        return self.model.predict(X)

class MLP():
    def __init__(self,xshape):
        self.model = self.buildModel(xshape)

    def buildModel(self,xshape):
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(xshape,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1))
        ##opt = keras.optimizers.Adam(learning_rate=0.01)
        # optimizer='rmsprop'

        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

        return model

    def fit(self,X,y):
        self.model.fit(X, y,epochs=250, batch_size=4, verbose=1)

    def predict(self,X):
        pass

    def savemodel(self):
        self.model.save('model.h5')