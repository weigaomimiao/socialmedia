#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: hierarCnn1d.py
@time: 2020/12/3 13:53
@desc:
'''
from .mlp import MLP
from .cnn1d import CCNN1d
class CMLP():
    def __init__(self,xshape,savemodel=True,basemodel='mlp'):
        self.model_low = MLP(xshape,savemodel)
        self.model_middle = MLP(xshape,savemodel)
        self.model_high = MLP(xshape, savemodel)

    def fit(self,X,y):
        pass

    def predict(self,X):
        pass

    def savemodel(self):
        pass

    def loadModel(self):
        pass