#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: savedModel.py
@time: 2020/11/29 17:46
@desc: all kinds of models, the final version of savedModel is also included.
'''
from .mlp import MLP
from .rf import RF


class Model():
    def __init__(self,xshape,modeltype='mlp'):
        self.model = self.initmodel(modeltype,xshape)

    def initmodel(self,modeltype,xshape):
        if modeltype=='mlp':
            return MLP(xshape)
        elif modeltype=='rf':
            return RF()
        return None

    def fit(self,X,y):
        self.model.fit(X,y)

    def predict(self,X):
        return self.model.predict(X)

    def savemodel(self):
        self.model.savemodel()

    def loadModel(self):
        return self.model.loadModel()

