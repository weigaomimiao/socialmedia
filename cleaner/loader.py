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
    def __init__(self,kbest=15,notPickfields=None,featuresEng='num+onehot',
                 dropOutlierRatio=0.25,discreteMethod=None,imgFeaDim=128):
        # check setting parameters
        if featuresEng not in ['num+onehot','box+onehot']:
            print('Please set featuresEng legally, only num+onehot and box+pnehot are available')
        if featuresEng=='box+onehot':
            if discreteMethod not in ['interval','frequency','DT']:
                print('Please set discreteMethod legally, only interval, frequency and DT are available')
        elif featuresEng=='num+onehot':
            if discreteMethod is not None:
                print('Please set discreteMethod as None because you are requiring numerical features')
        if dropOutlierRatio>0 and dropOutlierRatio <0.5: # control so that one cannot drop more than 50% of data as outliers.
            # used when Cleaner.deal_outlier()
            self.dropOutlierRatio_ = dropOutlierRatio
        if notPickfields is None:
            self.notPickfields_ = ['id', 'uname', 'url', 'covImgStatus', 'verifStatus', 'textColor', 'pageColor', 'themeColor',
                      'isViewSizeCustom',  'location', 'isLocVisible','creatTimestamp','utcOffset_hour','account_last','creatTimestamp_month',
                      'uTimeZone', 'numFollowers', 'numPeopleFollowing', 'numStatUpdate', 'numDMessage',
                      'avgvisitPerSecond','avgvisitPerSecondLog', 'avgClick', 'profileImg',
                       'numPLikes','hasUrl','textClass','pageClass','themeClass'] # features that won't be picked

        # 'utcOffset','creatTimestamp_year','creatTimestamp_month','avgClickLog','numFollowersLog','numStatUpdateLog'
        else:
            self.notPickfields_ = notPickfields
        # the original log numerical features won't be picked
        self.notPickfieldsNum_ = ['avgClickLog', 'numDMessageLog', 'numStatUpdateLog','numFollowersLog', 'numPeopleFollowingLog']
        #  'creatTimestamp_days',
        self.normFields = ['mes+visit','utcOffset','creatTimestamp_year','numStatUpdateLog','avgClickLog',
                           'numFollowersLog','numDMessageLog','numPeopleFollowingLog']+[str(i) for i in range(imgFeaDim)]
        self.method_ = featuresEng.split('+')[0]
        self.discreteMethod_ = discreteMethod
        if self.method_=='box':
            # if box features are required, drop the origin numerical features.
            # self.notPickfields_ += self.notPickfieldsNum_
            pass
        self.kbest = kbest

        # Initialize tools
        # Cleaner is initialized with method 'box' or 'num
        self.cleaner = Cleaner(method=self.method_,
                               dropOutlierRatio=self.dropOutlierRatio_,
                               discreteMethod=self.discreteMethod_)
        self.standar = Standardize()  # for standardization
        if(kbest>0):
            self.selector = FeaSelector(kbest) # for selecting features

        # processing data
        self.process()

    def process(self):
        df_train = self.loadData('train')
        df_test = self.loadData('test')
        self.testId = df_test['id'].values
        # set y, expand dims for model
        trainY = df_train['numPLikes'].values
        trainLogY = [np.log10(1.5 + i) for i in trainY]
        # trainLogY = self.dealyOutlier(trainLogY)

        df_trainX, df_testX = None,None
        if self.method_=='box': # directly take picked df as features, no need for normalization
            if self.discreteMethod_=='DT':
                df_trainX, df_testX = self.cleandateforboxing(
                    df_train=df_train.iloc[:, :-1], df_test=df_test,train_y=trainLogY)
            else:
                df_trainX, df_testX = self.cleanData(df_train.iloc[:, :-1], df_test,trainLogY)
            self.trainX, self.testX = df_trainX.values, df_testX.values
        else: # normalizing data only when one take numerical features.
            df_trainX, df_testX = self.cleanData(df_train.iloc[:, :-1], df_test,trainLogY)

        self.trainX, self.testX = self.normalizeData(df_trainX, df_testX)
        # self.trainX, self.testX = df_trainX.values,df_testX.values

        # selecting features.
        if (self.kbest > 0):
            self.trainX, self.testX = self.selectKBest(self.trainX, trainLogY, self.testX)
        # set y, expand dims for model
        self.trainLogY = np.expand_dims(trainLogY, axis=1)
        self.trainY = np.expand_dims(trainY, axis=1)
        del df_train, df_test, df_trainX, df_testX

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

    def cleanData(self,df_train,df_test,trainLogY):
        '''
        Clean train and test together rather than one by one
        :param df_train:
        :param df_test:
        :return:
        '''
        train_size = df_train.shape[0]
        df = pd.concat([df_train,df_test],ignore_index=True)

        df_new = self.cleaner.cleanData(df,trainLogY)

        df_new.to_csv('%s/data/X_all.csv'%getBasePath())
        # pick fields that are not in notPickfields_
        columns = df_new.columns.values
        column_picked = set(columns).difference(set(self.notPickfields_))
        df_picked = df_new[column_picked]
        # df_picked.to_csv('%s/data/X_picked.csv' % getBasePath())

        x_train = df_picked.iloc[:train_size,:]
        x_test = df_picked.iloc[train_size:, :]
        return x_train,x_test

    def normalizeData(self,df_trainX,df_testX):
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

    def cleandateforboxing(self,df_train,df_test,train_y):
        train_size = df_train.shape[0]
        df = pd.concat([df_train, df_test], ignore_index=True)

        df_new = self.cleaner.cleanDateBox(df,train_y)
        pickStartInd = self.cleaner.pickIndStart

        columns = df_new.columns.values
        column_picked = set(columns).difference(set(self.notPickfields_))

        # df_picked = df_new.iloc[:,pickStartInd:]
        df_picked = df_new[column_picked]
        x_train = df_picked.iloc[:train_size, :]
        x_test = df_picked.iloc[train_size:, :]
        return x_train, x_test

    def dealyOutlier(self,y):
        df_y = pd.DataFrame({'y':y})
        qu_high = df_y['y'].quantile(q=1 - self.dropOutlierRatio_/2)  # say 0.75
        qu_low = df_y['y'].quantile(q=self.dropOutlierRatio_/2)  # say 0.25
        value = qu_high - qu_low
        top = qu_high + 1.5 * value
        bottom = qu_low - 1.5 * value
        df_y['y'].where(cond=(df_y['y'] < top), other=top, inplace=True)
        df_y['y'].where(cond=(df_y['y'] > bottom), other=bottom, inplace=True)
        return df_y['y'].values