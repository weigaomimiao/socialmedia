#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: main.py
@time: 2020/11/29 18:20
@desc: main function for running savedModel
'''
from cleaner import BuildDataset
from models import Model,XGBoost
from models import FineTuning
from util import saveResult, trainvalLossPlot
from sklearn.model_selection import learning_curve

if __name__ == "__main__":
    # training set and test set building, select kbest feature
    builder = BuildDataset(kbest=-1)
    X, Y,Ylog,testX, testId, stander = builder.getData()

    # fit savedModel with part data and evaluation
    xshape = X.shape[1]
    model = XGBoost()
    # from xgboost import XGBRegressor
    # model = XGBRegressor(n_jobs=-1,n_estimators=500,learning_rate=0.05,min_child_weight=5,max_depth=6,random_state=0)
    # train_sizes, train_scores, test_scores, fit_times, _ = \
    #     learning_curve(model, X, Ylog, cv=15, n_jobs=-1,
    #                    train_sizes=[0.7,0.8,0.9,1],
    #                    return_times=True,shuffle=True,scoring='neg_mean_squared_error')
    # grid search
    param_grid = {'max_depth': [4, 5, 6, 7, 8], 'learning_rate': [1e2, 1e3], 'n_estimators': [200, 300, 400, 500]}
    # param_grid = None
    fitter = FineTuning(model,cv=10,param_grid=param_grid)
    model_best,params, best_score = fitter.finetuning(X,Ylog)
    print('best_params')
    print(params)
    print('best_score: %.3f'%best_score)
    model_best.savemodel()

    # trainvalLossPlot(train_scores,test_scores,model_name)

    # predict
    ypred = model_best.predict(testX)
    ypred = stander.inverse_log10_y(ypred)

    # save prediction to csv
    saveResult(testId,ypred)




