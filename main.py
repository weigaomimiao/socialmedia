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
from models import Model
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from util import saveResult, trainvalLossPlot
from sklearn.model_selection import learning_curve
from models import FineTuning

if __name__ == "__main__":
    # training set and test set building, select kbest feature
    builder = BuildDataset(kbest=-1)
    X, Y,Ylog,testX, testId, stander = builder.getData()

    # fit savedModel with part data and evaluation
    xshape = X.shape[1]
    model_name = 'xgboost'
    # model = Model(modeltype=model_name,xshape=xshape)
    from xgboost import XGBRegressor
    model = XGBRegressor(n_jobs=-1,n_estimators=500,learning_rate=0.05,min_child_weight=5,max_depth=6,random_state=0)
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(model, X, Ylog, cv=15, n_jobs=-1,
                       train_sizes=[0.7,0.8,0.9,1],
                       return_times=True,shuffle=True,scoring='neg_mean_squared_error')
    # kf = KFold(n_splits=15,random_state=None, shuffle=True)
    # kf.get_n_splits(X)
    # errlist = []
    # errtmp = 999999.
    # model_best, score = FineTuning(model,cv=15)
    # you can pass param_grid here or set the default grid in class

    # for train_index, val_index in kf.split(X):
    #     X_train, X_val = X[train_index,:], X[val_index,:]
    #     y_train, y_val = Ylog[train_index], Y[val_index]
    #     model.fit(X_train,y_train)
    #     # predict
    #     val_ypred = model.predict(X_val)
    #     val_ypred = stander.inverse_log10_y(val_ypred)
    #     # evaluation
    #     tmp = mean_squared_log_error(val_ypred,y_val)
    #     if(tmp<errtmp):
    #         errtmp = tmp
    #     print(tmp)
    #     errlist.append(tmp)
    # model.savemodel()

    trainvalLossPlot(train_scores,test_scores,model_name)

    # refit whole training set
    model.fit(X, Ylog)
    # savedModel = model.loadModel()
    ypred = model.predict(testX)
    ypred = stander.inverse_log10_y(ypred)

    # save prediction to csv
    saveResult(testId,ypred)




