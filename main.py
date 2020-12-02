#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: main.py
@time: 2020/11/29 18:20
@desc: main function for running model
'''
from clean import BuildDataset
from model import Model
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from util import saveResult

if __name__ == "__main__":
    # training set and test set building
    builder = BuildDataset()
    X, Y, testX, testId, stander = builder.getData()
    # trainX,trainY,evalX,evalY = train_test_split(X,Y,test_size=0.3,random_state=420)

    # fit model with part data and evaluation
    xshape = X.shape[1]
    model = Model(modeltype='mlp',xshape=xshape,savemodel=True)
    kf = KFold(n_splits=15,random_state=None, shuffle=False)
    kf.get_n_splits(X)
    errlist = []
    errtmp = 999999.
    for train_index, val_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", val_index)
        X_train, X_val = X[train_index,:], X[val_index,:]
        y_train, y_val = Y[train_index], Y[val_index]
        model.fit(X_train,y_train)
        # predict
        val_ypred = model.predict(X_val)
        # val_ypred = stander.inverse_standarde_y(eval_ypred)
        # evaluation
        tmp = mean_squared_log_error(val_ypred,y_val)
        if(tmp<errtmp):
            errtmp = tmp
            model.savemodel()
        print(tmp)
        errlist.append(tmp)

    # refit whole training set
    # model = Model('mlp')
    # model.fit(X, Y)
    ypred = model.predict(testX)
    # ypred = stander.inverse_standarde_y(ypred)

    # save prediction to csv
    saveResult(testId,ypred)




