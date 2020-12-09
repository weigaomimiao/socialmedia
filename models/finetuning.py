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

class FineTuning():
    def __init__(self,model,cv=10,param_grid=None):
        if param_grid is None:
            if model.__class__.__name__=='XGBoost':
                # param_grid = {'max_depth':[4,5,6,7,8],'learning_rate':[1e2,1e3],'n_estimators':[200,300,400,500]}
                param_grid = {'max_depth': [ 6, 7], 'learning_rate': [1e2, 1e3],
                              'n_estimators': [ 500]}
            elif model.__class__.__name__=='MLP':
                param_grid = {'epoch': [50, 100, 150, 200], 'learning_rate': [1e2, 1e3]}
        self.fitter = GridSearchCV(estimator=model,param_grid=param_grid,scoring='neg_mean_absolute_error',cv=cv)

    def finetuning(self,X,y):
        '''
        Fine tuning using GridSearchCV
        :return: mdoel, traning and validation scores
        '''
        model = self.fitter.fit(X,y)
        params = model.best_params_
        score = model.best_score_
        # scores = np.array(scores).reshape(-1, len(Gammas))
        return model.best_estimator_, params,score
