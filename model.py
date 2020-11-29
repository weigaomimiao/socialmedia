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
import tensorflow as tf
class Model():
    def __init__(self,modeltype):
        self.model = self.initmodel(modeltype)

    def initmodel(self,modeltype):
        if modeltype=='SVR':
            return SVR()
        return None

    def fit(self,X,y):
        self.model.fit(X,y)

    def predict(self,X):
        return self.model.predict(X)

