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

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import math
import util

from .ProfileImgNet import ProfileCNN,ImgLoader
class Cleaner():
    '''
    Only for cleaning data x, y is handled in BuildDataset()
    '''
    def __init__(self,method='num',dropOutlierRatio=0.25,discreteMethod='interval'):
        self.df = None
        # self.pickIndStart = -1 # the start index of picking fields
        self.method_ = method # indicate whether to take numerical features or cut to bins
        self.dropOutlierRatio_ = dropOutlierRatio
        self.discreteMethod_ = discreteMethod
        self.train_size = 7500
        self.test_size = 2500
        self.imgSize = (32,32,3)
        self.yBinsNum = 10

    def cleanData(self,df):
        self.df = df
        del df
        self.fillna()
        self.outlier_dealing()
        self.buildFeatures()
        # self.outlier_dealing()
        self.onehotencode() # encode discrete features
        self.rescaleData()  # take logarithmic of numerical features
        if self.method_=='box' and self.discreteMethod_!='DT':
            # if need boxing features, do cutting
            self.cut2box() # cut continuous data by pd.cut or pd.qcut
        return self.df

    def rescaleData(self):
        self.df['numFollowersLog'] = self.df['numFollowers'].map(lambda x: np.log10(1.5+x))
        self.df['numPeopleFollowingLog'] = self.df['numPeopleFollowing'].map(lambda x: np.log10(1.5+x))
        self.df['numStatUpdateLog'] = self.df['numStatUpdate'].map(lambda x: np.log10(1.5+x))
        self.df['numDMessageLog'] = self.df['numDMessage'].map(lambda x: np.log10(1.5+x))
        self.df['avgClickLog'] = self.df['avgClick'].map(lambda x: np.log10(1.5+x))

    def buildFeatures(self):
        self.classifyColor()
        self.extractImg()
        self.extractLoc()

        self.df['hasUrl'] = self.df['url'].isnull()
        self.df['hasUrl'] = self.df['hasUrl'].map(str)
        # feature starting ID, do put the line below after hasUrl
        self.pickIndStart = self.df.shape[1]

        self.df['utcOffset_hour'] = self.df['utcOffset'] / 3600
        create_year = self.df['creatTimestamp'].apply(lambda x: x.split()[-1]).tolist()
        create_year = [int(i) for i in create_year]
        self.df["creatTimestamp_year"] = create_year

    def onehotencode(self):
        self.df['isLocVisible'] = self.df['isLocVisible'].str.lower()
        self.df['covImgStatus'].fillna('Set', inplace=True)
        category = self.df.loc[:,
                   ['hasUrl', 'covImgStatus', 'verifStatus', 'isViewSizeCustom', 'isLocVisible', 'uLanguage','textClass','pageClass','themeClass']]
        category_df = pd.get_dummies(category, dummy_na=True)
        self.df = pd.concat([self.df,category_df],axis=1,ignore_index=False)

    def extractLoc(self):
        # extract features from field "location"
        pass

    def extractImg(self):
        # before extracting features, you can change epochs, yclassNum and feaDim in ProfileImgNet.py
        # After that, you will get a model under /savedModel, which is named as 'cnn-pfImg.h5'

        # extract features from profile images
        # 1) take image names
        # imgNameList = self.df['profileImg'].values
        #
        # # 2) load images, some don't exist
        # imgTrainLoader = ImgLoader(imgNameList[:self.train_size],tasktype='train',path="%s/data/%s_profile_images/profile_images_%s")
        # imgTestLoader = ImgLoader(imgNameList[self.train_size:], tasktype='test',path="%s/data/%s_profile_images/profile_images_%s")
        #
        # imgTrain,trainImgExisInd = imgTrainLoader.loadImgs()
        # imgTest,testImgExisInd = imgTestLoader.loadImgs()
        # print(np.invert(np.array(testImgExisInd)).sum())
        #
        # # 3) load model
        # extractor = ProfileCNN(xshape=self.imgSize, yclassNum=self.yBinsNum,isRelativePath=False) # need to be the same as what you pass while training model
        # extractor.loadModel()
        #
        # fea_train_part = extractor.extracFeas(imgTrain)
        # del imgTrain
        #
        # fea_test_part = extractor.extracFeas(imgTest)
        # del imgTest

        fea_train = pd.read_csv('%s/data/train-img-feas.csv'%util.getBasePath(),index_col=0)
        fea_test = pd.read_csv('%s/data/test-img-feas.csv' % util.getBasePath(),index_col=0)

        df_imgs = pd.concat([fea_train,fea_test],axis=0,ignore_index=True)

        self.df = pd.concat([self.df,df_imgs],axis=1)
        del df_imgs
        print('get images done')

    def classifyColor(self):
        '''
        cut color into color bins, and new feature: textClass, pageClass, themeClass
        :return:
        '''
        self.df["textColor"] = self.df["textColor"].fillna('1fc2de')  # fill with mode
        self.df["pageColor"] = self.df["pageColor"].fillna('ffffff')
        self.df["themeColor"] = self.df["themeColor"].fillna('000000')
        dftextr = self.df["textColor"].map(lambda x: int(str(x)[0:2], 16))
        dftextg = self.df["textColor"].map(lambda x: int(str(x)[2:4], 16))
        dftextb = self.df["textColor"].map(lambda x: int(str(x)[4:6], 16))

        dftextr = np.expand_dims(dftextr, axis=1)
        dftextg = np.expand_dims(dftextg, axis=1)
        dftextb = np.expand_dims(dftextb, axis=1)
        text_vec = np.concatenate((dftextr, dftextg, dftextb), axis=1)
        kmeans_text = KMeans(n_clusters=4).fit(text_vec)

        dfpager = self.df["pageColor"].map(lambda x: int(str(x)[0:2], 16))
        dfpageg = self.df["pageColor"].map(lambda x: int(str(x)[2:4], 16))
        dfpageb = self.df["pageColor"].map(lambda x: int(str(x)[4:6], 16))
        dfpager = np.expand_dims(dfpager, axis=1)
        dfpageg = np.expand_dims(dfpageg, axis=1)
        dfpageb = np.expand_dims(dfpageb, axis=1)
        page_vec = np.concatenate((dfpager, dfpageg, dfpageb), axis=1)
        kmeans_page = KMeans(n_clusters=4).fit(page_vec)

        dfthemer = self.df["themeColor"].map(lambda x: int(str(x)[0:2], 16))
        dfthemeg = self.df["themeColor"].map(lambda x: int(str(x)[2:4], 16))
        dfthemeb = self.df["themeColor"].map(lambda x: int(str(x)[4:6], 16))
        dfthemer = np.expand_dims(dfthemer, axis=1)
        dfthemeg = np.expand_dims(dfthemeg, axis=1)
        dfthemeb = np.expand_dims(dfthemeb, axis=1)
        theme_vec = np.concatenate((dfthemer, dfthemeg, dfthemeb), axis=1)
        kmeans_theme = KMeans(n_clusters=5).fit(theme_vec)

        self.df['textClass'] = [str(int(i)) for i in kmeans_text.labels_]
        self.df['pageClass'] = [str(int(i)) for i in kmeans_page.labels_]
        self.df['themeClass'] = [str(int(i)) for i in kmeans_theme.labels_]

    def fillcategory(self):
        # feature use: verifystatus, hasurl, islocvisible, numstatsupdate
        data_set = self.df.loc[:, ['verifStatus', 'url', 'isLocVisible', 'numStatUpdate', 'category']]
        self.isLocVisible_lowercase()
        data_set['url'] = data_set['url'].isnull()
        data_set['url'] = data_set['url'].replace([True, False], [False, True])

        data_set['temp'] = range(0, self.df.shape[0]) # avoid bug when select train_set

        # seprate train_set,test_set
        test_set = data_set[data_set['category'] == ' ']
        test_x = test_set.loc[:, ['verifStatus', 'url', 'isLocVisible', 'numStatUpdate']]

        temp = data_set.append(test_set)
        train_set = temp.drop_duplicates(keep=False)
        train_x = train_set.loc[:, ['verifStatus', 'url', 'isLocVisible', 'numStatUpdate']]
        train_y = train_set['category']

        # transform str to number
        trans1 = LabelEncoder()
        train_x['isLocVisible'] = trans1.fit_transform(train_x['isLocVisible'])
        test_x['isLocVisible'] = trans1.transform(test_x['isLocVisible'])

        trans2 = LabelEncoder()
        train_x['url'] = trans2.fit_transform(train_x['url'])
        test_x['url'] = trans2.transform(test_x['url'])

        trans3 = LabelEncoder()
        train_x['verifStatus'] = trans3.fit_transform(train_x['verifStatus'])
        test_x['verifStatus'] = trans3.transform(test_x['verifStatus'])

        # normalize number follower
        tran4 = MinMaxScaler()
        train_x = tran4.fit_transform(train_x)
        test_x = tran4.fit_transform(test_x)

        #use knn to predict score>0.65
        model = KNeighborsClassifier(n_neighbors=15, weights='uniform')
        model.fit(train_x, train_y)
        res = model.predict(test_x)

        test_set = data_set[data_set['category'] == ' ']
        test_x = test_set.loc[:, ['verifStatus', 'url', 'isLocVisible', 'numStatUpdate']]
        test_x['category'] = res

        #replace ' ' in category by predict result
        filled_data = pd.concat([train_set, test_x], axis=0)
        filled_data = filled_data.sort_index()
        self.df['category'] = filled_data['category']

        # change outlier to 1/4point +\- (1.5* 1/4high \ 1/4low)
    def outlier_dealing(self):
        numeric_columns = self.df.loc[:,['numFollowers', 'numPeopleFollowing', 'numStatUpdate', 'numDMessage', 'avgClick']]
        for column in numeric_columns:
            set = self.df[column]
            qu_high = set.quantile(q=1-self.dropOutlierRatio_) # say 0.75
            qu_low = set.quantile(q=self.dropOutlierRatio_) # say 0.25
            value = qu_high - qu_low
            top = qu_high + 1.5 * value
            bottom = qu_low - 1.5 * value
            set.where(cond=(set < top), other=top, inplace=True)
            set.where(cond=(set > bottom), other=bottom, inplace=True)
            self.df[column] = set

    def fillavgclick(self):
        # feature using: verifStatus,hasurl,category
        data_set = self.df.loc[:, ['verifStatus', 'url', 'category', 'avgClick', 'numPeopleFollowing', 'numFollowers']]
        data_set['url'] = data_set['url'].isnull()
        data_set['url'] = data_set['url'].replace([True, False], [False, True])

        # seprate train_set,test_set
        test_set = data_set[data_set['avgClick'].isnull()]
        test_x = test_set.loc[:, ['verifStatus', 'url', 'category', 'numPeopleFollowing', 'numFollowers']]

        temp = data_set.append(test_set)
        train_set = temp.drop_duplicates(keep=False)
        train_x = train_set.loc[:, ['verifStatus', 'url', 'category', 'numPeopleFollowing', 'numFollowers']]
        train_y = train_set['avgClick']

        # transform str to number
        trans1 = LabelEncoder()
        train_x['verifStatus'] = trans1.fit_transform(train_x['verifStatus'])
        test_x['verifStatus'] = trans1.transform(test_x['verifStatus'])

        trans2 = LabelEncoder()
        train_x['url'] = trans2.fit_transform(train_x['url'])
        test_x['url'] = trans2.transform(test_x['url'])

        trans3 = LabelEncoder()
        train_x['category'] = trans3.fit_transform(train_x['category'])
        test_x['category'] = trans3.transform(test_x['category'])

        # normalize number follower
        tran4 = MinMaxScaler()
        train_x = tran4.fit_transform(train_x)
        new_test_x = tran4.fit_transform(test_x)

        # use ridge regression predict score>0.14
        model2 = Ridge(alpha=1)
        model2.fit(train_x, train_y)
        res = model2.predict(new_test_x)

        test_set = data_set[data_set['avgClick'].isnull()]
        test_x = test_set.loc[:, ['verifStatus', 'url', 'category', 'numPeopleFollowing', 'numFollowers']]
        test_x['avgClick'] = res

        # replace NaN in avgClick by predicton result
        filled_data = pd.concat([train_set, test_x], axis=0)
        filled_data = filled_data.sort_index()
        self.df['avgClick'] = filled_data['avgClick']


    def isLocVisible_lowercase(self):
        self.df['isLocVisible'] = self.df['isLocVisible'].str.lower()

    def fillna(self):
        # filling all blanks, can call like fillLoc etc.
        # self.df['covImgStatus'].fillna('unknown', inplace=True)
        # fill utcOffset
        self.df['utcOffset'].fillna(0,inplace=True)
        self.fillLoc()
        self.fillcategory()
        self.fillavgclick()

    def fillLoc(self):
        # filling?
        pass

    def cut2box(self):
        '''
        Discretize numerical continuous data, use pd.cut or pd.qcut
        :return:
        '''
        interval_log = 0.05
        def getNums(fieldName):
            # set the interval of log values while cutting, then calculate the number of bins
            a = math.ceil((self.df[fieldName].max()-self.df[fieldName].min())/interval_log)
            return a
        binsDict = {
            'avgClickLog':getNums('avgClickLog'),
            'numDMessageLog':getNums('numDMessageLog'),
            'numStatUpdateLog': getNums('numStatUpdateLog'),
            'numFollowersLog':getNums('numFollowersLog'),
            'numPeopleFollowingLog':getNums('numPeopleFollowingLog')
        }
        cols = list(binsDict.keys())
        numerical = self.df[cols].copy(deep=True)
        for c in cols:
            if self.discreteMethod_=='interval':
                numerical[c] = pd.cut(numerical[c], binsDict[c])
            elif self.discreteMethod_=='frequency':
                numerical[c] = pd.qcut(numerical[c], binsDict[c],duplicates='drop')
        category = pd.get_dummies(numerical, dummy_na=True)
        self.df = pd.concat([self.df, category], axis=1, ignore_index=False)

    def boxing(self,train_y): # will be called if discreteMethod=='DT'
        numerical = self.df.loc[:,
                    ['avgClickLog', 'numDMessageLog', 'numStatUpdateLog',
                     'numFollowersLog', 'numPeopleFollowingLog']]
        for column in numerical:
            boundary = []
            box = DecisionTreeRegressor(min_samples_leaf=0.05)
            x = np.array(numerical.loc[:, [column]])
            y = np.array(train_y)
            box.fit(x[:x.shape[0]], y)

            n_nodes = box.tree_.node_count  # 决策树的节点数
            children_left = box.tree_.children_left  # node_count大小的数组，children_left[i]表示第i个节点的左子节点
            children_right = box.tree_.children_right  # node_count大小的数组，children_right[i]表示第i个节点的右子节点
            threshold = box.tree_.threshold  # node_count大小的数组，threshold[i]表示第i个节点划分数据集的阈值

            for i in range(n_nodes):
                if children_left[i] != children_right[i]:  # 非叶节点
                    boundary.append(threshold[i])

            min_x = x.min()
            max_x = x.max() + 1e-3
            boundary.sort()
            boundary = [min_x] + boundary + [max_x]
            self.df[column] = pd.cut(x=self.df[column], bins=boundary, right=True, labels=range(0, len(boundary) - 1))
        category = self.df.loc[:,
                       ['avgClickLog', 'numDMessageLog', 'numStatUpdateLog',
                        'numFollowersLog', 'numPeopleFollowingLog']]
        category_df = pd.get_dummies(category, dummy_na=True)
        self.df = pd.concat([self.df, category_df], axis=1, ignore_index=False)

    def cleanDateBox(self,df,train_y):
        self.df = df
        del df
        self.fillna()
        self.outlier_dealing()
        self.buildFeatures()
        self.onehotencode()  # encode discrete features
        self.rescaleData()  # take logarithmic of numerical features
        self.boxing(train_y)
        return self.df
class Standardize():
    def __init__(self):
        self.standrdX = StandardScaler()

    def fit(self,X):
        self.standrdX.fit(X)

    def transform(self,X):
        return self.standrdX.transform(X,copy=True)

    def inverse_log10_y(self,Y):
        ypred = np.power(10,Y)-1.5 # cause y is rescaled as log(1+x)
        ypred = np.ceil(ypred).astype(int)
        return ypred
class FeaSelector():
    def __init__(self,k):
        self.selector = SelectKBest(mutual_info_regression,k=k)

    def fit(self,X,y):
        self.selector.fit(X, y)
    def select(self,X):
        X_new = self.selector.transform(X)
        return X_new
