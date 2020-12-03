#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: rf.py
@time: 2020/12/3 13:50
@desc:
'''
import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from util import getBasePath

class RF(): # random forest
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100,
                                           n_jobs=-1,
                                           bootstrap=True,
                                           max_depth=5,
                                           random_state=0)

    def fit(self, X, y):
        ytmp = y
        if len(ytmp.shape)>1:
            ytmp = np.squeeze(ytmp,axis=1)
        self.model.fit(X, ytmp)
        # for r^2 score, the bigger the better
        print("R^2 Training score: {%.2f}"%(self.model.score(X,ytmp)))

    def predict(self, X):
        ypred = self.model.predict(X)
        return ypred

    def savemodel(self):
        joblib.dump(self.model,'%s/savedModel/model-rf.joblib' % getBasePath(),compress=0)

    def loadModel(self):
        return joblib.load('%s/savedModel/model-rf.joblib' % getBasePath())