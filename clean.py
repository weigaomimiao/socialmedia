#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: clean.py
@time: 2020/11/25 19:40
@desc: take a look of data and clean it
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import util
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

sns.set()

class Loader():
    def __init__(self,pickfields=None):
        self.df = None
        self.indexlist_ = ['id','uname','url','covImgStatus','verifStatus','textColor','pageColor','themeColor','isViewSizeCustom','utcOffset','location','isLocVisible','uLanguage','creatTimestamp','uTimeZone','numFollowers','numPeopleFollowing','numStatUpdate','numDMessage','category','avgvisitPerSecond','avgClick','profileImg','numPLikes']
        self.stander = StandardScaler() # for standardization if needed
        if pickfields is None:
            self.pickFields_ = ['avgvisitPerSecond']
        else:
            self.pickFields_ = pickfields

    def loadData(self,dataset):
        basepath = util.getBasePath()
        astr = "%s/data/%s.csv"
        if dataset not in ['train','test']:
            print("Invalid dataset type, only train and test are supported")
            return ""
        filename = astr % (basepath, dataset)
        self.df = pd.read_csv(filename)
        self.df.columns = self.indexlist_

        self.fillna()
        self.recodeData()

        if(dataset=='train'):
            return self.df[self.pickFields_]
        else:
            return self.df[self.pickFields_[:-2]],self.df['id']

    def recodeData(self):
        # field Personal URL: 1 for not null, 0 for null
        self.df['hasUrl'] = self.df['url'].isnull()

        # encode conImgStatus
        le = LabelEncoder()
        le.fit(self.df['covImgStatus'].unique())
        distr = le.classes_
        di = {distr[0]: 0, distr[1]: 1, distr[2]: 2}
        self.df.replace({"covImgStatus": di})

        self.extractImg()
        self.extractLoc()

    def extractLoc(self):
        # extract features from field "location"
        self.df['newName'] = "blabla"

    def extractImg(self):
        # extract features from profile image
        self.df['newName'] = "blabla"

    def fillna(self):
        # filling all blanks, can call like fillLoc etc.
        self.df['covImgStatus'].fillna('unknown', inplace=True)
        # fill location
        self.fillLoc()

    def fillLoc(self):
        # filling?
        pass

class Standardize():
    def __init__(self):
        self.standrdX = StandardScaler()
        self.standrdY = StandardScaler()

    def fit(self,X,Y):
        self.standrdX.fit(X)
        self.standrdX.fit(Y)

    def transform(self,X,Y=None):
        if Y is None:
            return self.standrdX.transform(X,copy=True)
        else:
            return self.standrdX.transform(X,copy=True),self.standrdY.transform(Y,copy=True)

    def inverse_standarde_y(self,Y):
        return self.standrdY.inverse_transform(Y)

class BuildDataset():
    def __init__(self):
        self.standar = Standardize()
        loader = Loader()
        df_train = loader.loadData('train')
        df_test,testId = loader.loadData('test')
        trainX, trainY = df_train.iloc[:, :-2].values, df_train.iloc[:, -1].values
        testX= df_test.values
        self.stander.fit(trainX, trainY)
        self.trainX,self.trainY= self.stander.transform(trainX, trainY)
        self.testX = self.stander.transform(testX)
        self.testId = testId

        return self.trainX,self.trainY,self.testX,testId,self.standar