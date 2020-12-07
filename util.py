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
import cv2
from sklearn.model_selection import learning_curve

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
    sns.set(font_scale=2.5)
    sns.heatmap(mat_corr, square=True, annot=True)
    # print(np.where((np.abs(mat_corr)>0.8).item() and (np.abs(mat_corr)<1).item()))
    plt.savefig("%s/data/corr.png"%getBasePath(),dpi=300)
    plt.close()

#======================  for showing image collections  ======================
def manyImgs(scale, imgarray):
    '''
    https://blog.csdn.net/qq_44703724/article/details/105613611
    make images in any amount in m rows and n cols.
    :param scale: scale factor
    :param imgarray:
    :return:
    '''
    rows = len(imgarray)         # the length of tuple or list
    cols = len(imgarray[0])      # if imgarray is list, return #channels of the 1st image of it. If tuple, return the length of the 1st list in tuple
    # print("rows=", rows, "cols=", cols)

    # check whether imgarray[0] is list
    # If list，declare imgarray is a tuple, and show it vertically.
    rowsAvailable = isinstance(imgarray[0], list)

    # the width, height of the 1st img.
    width = imgarray[0][0].shape[1]
    height = imgarray[0][0].shape[0]
    # print("width=", width, "height=", height)

    # If you get a tuple
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                # iterate on the whole tuple, if the 1st one is an image, do nothing
                if imgarray[x][y].shape[:2] == imgarray[0][0].shape[:2]:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (0, 0), None, scale, scale)
                # resize others in the same size of the 1st image, by rescale factor
                else:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (imgarray[0][0].shape[1], imgarray[0][0].shape[0]), None, scale, scale)
                # If image is gray, change it in colors.
                if  len(imgarray[x][y].shape) == 2:
                    imgarray[x][y] = cv2.cvtColor(imgarray[x][y], cv2.COLOR_GRAY2BGR)

        # Create a blank img with the same size of the 1st img
        imgBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imgBlank] * rows   #the same size of the 1st img, blank
        for x in range(0, rows):
            # The xth list in tuple is shown horizontally.
            hor[x] = np.hstack(imgarray[x])
        ver = np.vstack(hor)   # All cols shown vertically
    # If you get a list
    else:
        # same transformation option
        for x in range(0, rows):
            if imgarray[x].shape[:2] == imgarray[0].shape[:2]:
                imgarray[x] = cv2.resize(imgarray[x], (0, 0), None, scale, scale)
            else:
                imgarray[x] = cv2.resize(imgarray[x], (imgarray[0].shape[1], imgarray[0].shape[0]), None, scale, scale)
            if len(imgarray[x].shape) == 2:
                imgarray[x] = cv2.cvtColor(imgarray[x], cv2.COLOR_GRAY2BGR)
        # make cols in horizon.
        hor = np.hstack(imgarray)
        ver = hor
    return ver

def collectImgs(imgNameList,m,n,labelInd,dataset):
    '''
    Fetch imgs according to imgNameList, and show in m rows and n cols
    :param imgNameList:
    :param m: m rows
    :param n: n cols
    :param labelInd: the label index
    :param dataset: train or test
    :return:
    '''
    from util import manyImgs
    imgs = []
    ih,iv=32,32
    img_blank = np.zeros((ih,iv,3),np.uint8)
    path = getBasePath() + '/../data/%s_profile_images/profile_images_%s/%s'
    # read imgs
    ind = 0
    for i in range(m):
        imgs_rows = []
        for j in range(n):
            if(ind<len(imgNameList)):
                imgName = imgNameList[ind]
                imgName = path % (dataset, dataset, imgName)
                if(not os.path.exists(imgName)):
                    imgs_rows.append(img_blank)
                else:
                    imgs_rows.append(cv2.imread(imgName))
                ind+=1
            else:
                imgs_rows.append(img_blank)
        imgs.append(imgs_rows)

    stackedimages = manyImgs(1, (imgs))

    cv2.namedWindow("collection-%s-%d"%(dataset,labelInd))
    cv2.imshow("collection-%s-%d"%(dataset,labelInd), stackedimages)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def trainvalLossPlot(trainloss,valloss,modelname):
    plt.figure('Learning Curve')
    n = len(trainloss)
    plt.plot(range(1,n+1),trainloss,color='blue',label='train')
    plt.plot(range(1, n + 1), valloss, color='orange',label='val')
    plt.title(modelname)
    plt.legend()
    plt.show()