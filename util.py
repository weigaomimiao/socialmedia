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
def getBasePath():
    return os.getcwd()

def saveResult(id,ypred):
    ypred = np.squeeze(np.ceil(ypred),axis=1)
    df_save = pd.DataFrame({'Id':id,'NumProfileLikes':ypred})
    df_save.to_csv('%s/data/submission.csv'%getBasePath(),index=False)
