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
from xgboost import XGBRegressor
import numpy as np
from util import getBasePath
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

class XGBoost(BaseEstimator, RegressorMixin): # XGBoost
    def __init__(self,n_estimators=500,learning_rate=0.05,min_child_weight=5,max_depth=6,random_state=0):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = XGBRegressor(objective ='reg:squarederror', n_jobs=-1,n_estimators=self.n_estimators,learning_rate=self.learning_rate,min_child_weight=self.min_child_weight,max_depth=self.max_depth,random_state=self.random_state)
    def fit(self, X, y):
        ytmp = y
        if len(ytmp.shape)>1:
            ytmp = np.squeeze(ytmp,axis=1)
        self.model.fit(X, ytmp)
        # for r^2 score, the bigger the better
        # print("R^2 Training score: {%.2f}"%(self.model.score(X,ytmp)))

    def predict(self, X):
        ypred = self.model.predict(X)
        return ypred

    def savemodel(self):
        joblib.dump(self.model,'%s/savedModel/model-xgboost.joblib' % getBasePath(),compress=0)

    def loadModel(self):
        return joblib.load('%s/savedModel/model-xgboost.joblib' % getBasePath())

    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "min_child_weight": self.min_child_weight,
                "max_depth": self.max_depth}

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