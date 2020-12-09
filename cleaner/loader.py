#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: loader.py
@time: 2020/12/6 21:30
@desc:
'''
import pandas as pd
import numpy as np
from util import getBasePath
from .clean import FeaSelector,Standardize,Cleaner

class BuildDataset():
    def __init__(self,kbest=15,notPickfields=None):
        self.standar = Standardize() # for standardization
        if notPickfields is None:
            # self.pickFields_ = ['utcOffset','creatTimestamp_year','avgClickLog','numDMessageLog','numStatUpdateLog','numFollowersLog','numPeopleFollowingLog', 'hasUrl','category','verifStatus','textClass','pageClass','themeClass','isLocVisible','uLanguage','isViewSizeCustom']
            self.notPickfields_ = ['id', 'uname', 'url', 'covImgStatus', 'verifStatus', 'textColor', 'pageColor', 'themeColor',
                      'isViewSizeCustom', 'utcOffset', 'location', 'isLocVisible', 'uLanguage', 'creatTimestamp',
                      'uTimeZone', 'numFollowers', 'numPeopleFollowing', 'numStatUpdate', 'numDMessage',
                      'category', 'avgvisitPerSecond', 'avgClick', 'profileImg', 'numPLikes','hasUrl']
        else:
            self.notPickfields_ = notPickfields
        self.kbest = kbest
        if(kbest>0):
            self.selector = FeaSelector(kbest) # for selecting features
        self.cleaner = Cleaner()
        self.normFields = ['utcOffset_hour','creatTimestamp_year','avgClickLog','numDMessageLog','numStatUpdateLog','numFollowersLog','numPeopleFollowingLog']

        df_train = self.loadData('train')
        df_test = self.loadData('test')
        self.testId = df_test['id'].values
        # set y, expand dims for model
        trainY = df_train['numPLikes'].values
        trainLogY = [np.log10(1.5 + i) for i in trainY]

        # clean
        df_trainX, df_testX = self.cleanData(df_train.iloc[:, :-1], df_test)
        # standard x and select kbest x
        self.trainX,self.testX = self.normalizeData(df_trainX,df_testX)
        if (kbest > 0):
            self.trainX,self.testX = self.selectKBest(self.trainX,trainLogY,self.testX)
        # set y, expand dims for model
        self.trainLogY = np.expand_dims(trainLogY, axis=1)
        self.trainY = np.expand_dims(trainY, axis=1)
        del df_train,df_test,df_trainX,df_testX

    def getData(self):
        return self.trainX,self.trainY,self.trainLogY,self.testX,self.testId,self.standar

    def loadData(self, dataset):
        indexlist_ = ['id', 'uname', 'url', 'covImgStatus', 'verifStatus', 'textColor', 'pageColor', 'themeColor',
                      'isViewSizeCustom', 'utcOffset', 'location', 'isLocVisible', 'uLanguage', 'creatTimestamp',
                      'uTimeZone', 'numFollowers', 'numPeopleFollowing', 'numStatUpdate', 'numDMessage',
                      'category', 'avgvisitPerSecond', 'avgClick', 'profileImg', 'numPLikes']
        basepath = getBasePath()
        astr = "%s/data/%s.csv"
        if dataset not in ['train', 'test']:
            print("Invalid dataset type, only train and test are supported")
            return ""
        filename = astr % (basepath, dataset)

        df = pd.read_csv(filename)
        if dataset == 'test':
            df.columns = indexlist_[:-1]
        else:
            df.columns = indexlist_
        return df

    def cleanData(self,df_train,df_test):
        '''
        Clean train and test together rather than one by one
        :param df_train:
        :param df_test:
        :return:
        '''
        train_size = df_train.shape[0]
        df = pd.concat([df_train,df_test],ignore_index=True)

        df_new = self.cleaner.cleanData(df)
        pickStartInd = self.cleaner.pickIndStart

        columns = df_new.columns.values
        column_picked = set(columns).difference(set(self.notPickfields_))

        # df_picked = df_new.iloc[:,pickStartInd:]
        df_picked = df_new[column_picked]
        x_train = df_picked.iloc[:train_size,:]
        x_test = df_picked.iloc[train_size:, :]
        return x_train,x_test

    def normalizeData(self,df_trainX,df_testX):
        len_norm = len(self.normFields)
        # fitting
        self.standar.fit(df_trainX[self.normFields].values)
        # transform
        trainX_part = self.standar.transform(df_trainX[self.normFields].values)
        # test set: standardization and feature selection
        testX_part = self.standar.transform(df_testX[self.normFields].values)

        # recombine data (normalized and unnormalized)
        columns = df_trainX.columns
        column_class = set(columns).difference(set(self.normFields))
        trainX = np.concatenate((trainX_part,df_trainX[column_class].values),axis=1)
        testX = np.concatenate((testX_part, df_testX[column_class].values), axis=1)

        return trainX,testX

    def selectKBest(self,trainX,trainLogY,testX):
        self.selector.fit(trainX, trainLogY)
        trainX_new = self.selector.select(trainX)
        testX_new = self.selector.select(testX)
        return trainX_new,testX_new