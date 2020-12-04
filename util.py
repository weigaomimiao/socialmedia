#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: util.py
@time: 2020/11/25 19:43
@desc: utilities: path...
'''
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def getBasePath():
    return os.getcwd()

def saveResult(id,ypred):
    if len(ypred.shape)>1:
        ypred = np.squeeze(ypred,axis=1)
    df_save = pd.DataFrame({'Id':id,'Predicted':ypred})
    df_save.to_csv('%s/data/submission.csv'%getBasePath(),index=False)

def matplot(df):
    sns.set()
    plt.figure(figsize=(50, 50))
    # columns = df.columns.values
    # print(columns)
    # columns_mat = np.tile(columns,(columns.shape[0],1))
    mat_corr = df.corr()
    sns.heatmap(mat_corr, square=True, annot=True)
    # print(np.where((np.abs(mat_corr)>0.8).item() and (np.abs(mat_corr)<1).item()))
    plt.savefig("%s/data/corr.png"%getBasePath(),dpi=500)
