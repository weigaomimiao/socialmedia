#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: siftProfileImg.py
@time: 2020/12/4 18:55
@desc:
'''
import cv2
from util import getBasePath,manyImgs,collectImgs
from sklearn.cluster import KMeans
import os
import numpy as np
class SIFTExtractor():
    def __init__(self, n_clusters=10,isRelativePath=True):
        self.sift_extractor = cv2.xfeatures2d_SIFT.create()
        if not isRelativePath:
            self.path = getBasePath() + '/../data/profile_images_%s/%s'
        else:
            self.path = getBasePath() + '/data/profile_images_%s/%s'
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=3724)
        self.nullImgInds = [] # keep tracking those with not existed profile images, int

    def extractFeas(self, imgNameList, dataset):
        features = []
        numNullCount = 0 # the amount of not existing images
        print("Those profile images that are not existed:")
        print(10*"-")
        for imgName in imgNameList:
            imgName = self.path % (dataset, imgName)

            img = np.zeros((32,32,3),np.uint8)

            if(not os.path.exists(imgName)): # for not existing images, fill with pure black image.
                numNullCount += 1
                # print(imgName)
            else:
            # assert os.path.exists(imgName)
                img = cv2.imread(imgName)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            kp, des = self.sift_extractor.detectAndCompute(gray_img, None)
            if des is None:
                des = np.zeros((1,128))
            reshape_feature = des.mean(axis=0).reshape((1,128))
            features.append(reshape_feature)
        print(numNullCount)
        print(10 * "-")
        features = np.squeeze(np.array(features),axis=1)
        return features

    def fit(self, imgNameList):
        features = self.extractFeas(imgNameList, 'train')
        input_x = features
        self.kmeans.fit(input_x)

    def classifyImgs(self, dataset, imgNameList):
        if dataset not in ['train', 'test']:
            print("Please set dataset for SIFTExtractor correctly, only train and test are available")

        if dataset == "train":
            return self.kmeans.labels_
        elif dataset == 'test':
            features = self.extractFeas(imgNameList, dataset)
            input_x = np.array(features)
            return self.kmeans.predict(input_x)

    # def testAImg(self):
    #     path = '%s/../data/train_profile_images/profile_images_train/0A2O8MC5RIG65CEA.png'%(getBasePath())
    #     image = cv2.imread(path)
    #     kp_image, _, des = self.sift_kp(image)
    #     print(image.shape, des.shape)
    #     cv2.namedWindow('train1', cv2.WINDOW_NORMAL)
    #     cv2.imshow('train1', kp_image)
    #     if cv2.waitKey(0) == 27:
    #         cv2.destroyAllWindows()

if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    df_train = pd.read_csv("%s/../data/train.csv" % getBasePath())
    df_test = pd.read_csv("%s/../data/test.csv" % getBasePath())
    indexlist_ = ['id', 'uname', 'url', 'covImgStatus', 'verifStatus', 'textColor', 'pageColor', 'themeColor',
                  'isViewSizeCustom', 'utcOffset', 'location', 'isLocVisible', 'uLanguage', 'creatTimestamp',
                  'uTimeZone', 'numFollowers', 'numPeopleFollowing', 'numStatUpdate', 'numDMessage', 'category',
                  'avgvisitPerSecond', 'avgClick', 'profileImg', 'numPLikes']
    df_train.columns = indexlist_
    df_test.columns = indexlist_[:-1]

    imgNamelist_train = df_train['profileImg'].values
    imgNamelist_test = df_test['profileImg'].values

    n_clusters = 10
    extractor = SIFTExtractor(n_clusters=10,isRelativePath=False)
    extractor.fit(imgNamelist_train)
    img_label_train = extractor.classifyImgs('train', imgNamelist_train)

    img_label_test = extractor.classifyImgs('test', imgNamelist_test)

    # collect img names in the same class
    imgs_labeled_train = [imgNamelist_train[img_label_train == i] for i in range(n_clusters)]
    print([len(i) for i in imgs_labeled_train])
    imgs_labeled_test = [imgNamelist_test[img_label_test == i] for i in range(n_clusters)]
    print([len(i) for i in imgs_labeled_test])
