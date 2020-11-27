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

sns.set()

def loadData(dataset):
    basepath = util.getBasePath()
    astr = "%s/data/%s.csv"
    if dataset not in ['train','test']:
        print("Invalid dataset type, only train and test are supported")
        return ""
    filename = astr % (basepath, dataset)
    df = pd.read_csv(filename)
    print(df.columns)
    abbrevIndex(df)
    return df

def abbrevIndex(df):
    indexList = ['id','uname','url','covImgStatus','verifStatus','textColor','pageColor','themeColor','isViewSizeCustom','utcOffset','location','isLocVisible','uLanguage','creatTimestamp','uTimeZone','numFollowers','numPeopleFollowing','numStatUpdate','numDMessage','category','avgvisitPerSecond','avgClick','profileImg','numPLikes']
    df.columns = indexList


def recodeData(df):
    # field Personal URL: 1 for not null, 0 for null
    df.url = ""



if __name__=="__main__":
    df_train = loadData('train')
    print(df_train.info())
    print(df_train.describe())