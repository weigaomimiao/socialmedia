#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: finetuning.py
@time: 2020/12/6 21:46
@desc:
'''
from sklearn.model_selection import GridSearchCV
import numpy as np
class FineTuning():
    def __init__(self,model,cv=10,param_grid=None):
        if param_grid is None:
            if model.__name__=='XGBoost':
                param_grid = {'max_depth':[4,5,6,7,8],'lr':[1e2,1e3],'n_estimators':[200,300,400,500]}
            elif model.__name__=='MLP':
                param_grid = {'epoch': [50, 100, 150, 200], 'lr': [1e2, 1e3]}
        self.fitter = GridSearchCV(estimator=model,param_grid=param_grid,scoring='mse',cv=cv)

    def finetuning(self,X,y):
        '''
        Fine tuning using GridSearchCV
        :return: mdoel, traning and validation scores
        '''
        model = self.fitter.fit(X,y)
        scores = model.grid_scores_
        # scores = np.array(scores).reshape(-1, len(Gammas))
        return model,scores
