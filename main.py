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
from cleaner.clean import BuildDataset
from models.model import Model
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from util import saveResult

if __name__ == "__main__":
    # training set and test set building, select kbest feature
    builder = BuildDataset(kbest=-1)
    # X, Y,Ynorm,testX, testId, stander = builder.getData()
    X, Y,Ylog,testX, testId, stander = builder.getData()
    # Ynorm is y after normalized, Y is the real integer data

    # fit savedModel with part data and evaluation
    xshape = X.shape[1]
    model = Model(modeltype='mlp',xshape=xshape)
    kf = KFold(n_splits=15,random_state=None, shuffle=False)
    kf.get_n_splits(X)
    errlist = []
    errtmp = 999999.
    for train_index, val_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", val_index)
        X_train, X_val = X[train_index,:], X[val_index,:]
        y_train, y_val = Ylog[train_index], Y[val_index]
        # y_train, y_val = Y[train_index], Y[val_index]

        model.fit(X_train,y_train)
        # predict
        val_ypred = model.predict(X_val)
        val_ypred = stander.inverse_log10_y(val_ypred)
        # evaluation
        tmp = mean_squared_log_error(val_ypred,y_val)
        if(tmp<errtmp):
            errtmp = tmp
            model.savemodel()
        print(tmp)
        errlist.append(tmp)

    # refit whole training set
    # savedModel = Model('mlp')
    # savedModel.fit(X, Y)
    model_fit = model.loadModel()
    ypred = model_fit.predict(testX)
    # ypred = stander.inverse_standarde_y(ypred)

    # save prediction to csv
    saveResult(testId,ypred)




