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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from util import saveResult

if __name__ == "__main__":
    # training set and test set building
    X,Y,testX,testY,testId,stander = BuildDataset()
    trainX,trainY,evalX,evalY = train_test_split(X,Y,test_size=0.3,random_state=420)

    # fit model with part data and evaluation
    model = Model('svr')
    model.fit(trainX,trainY)
    # predict
    eval_ypred = model.predict(evalX)
    eval_ypred = stander.inverse_standarde_y(eval_ypred)
    # evaluation
    print(mean_squared_error(eval_ypred,evalY))


    # refit whole training set
    model = Model('svr')
    model.fit(X, Y)
    ypred = model.predict(testX)
    ypred = stander.inverse_standarde_y(ypred)

    # save prediction to csv
    saveResult(testId,ypred)




