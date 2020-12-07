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
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
class Cleaner():
    '''
    Only for cleaning data x, y is handled in BuildDataset()
    '''
    def __init__(self):
        self.df = None
        self.pickIndStart = -1 # the start index of picking fields
        # self.indexlist_ = ['id','uname','url','covImgStatus','verifStatus','textColor','pageColor','themeColor','isViewSizeCustom','utcOffset','location','isLocVisible','uLanguage','creatTimestamp','uTimeZone','numFollowers','numPeopleFollowing','numStatUpdate','numDMessage','category','avgvisitPerSecond','avgClick','profileImg','numPLikes']
        # if pickfields is None:
        #     self.pickFields_ = ['utcOffset', 'creatTimestamp_year','numDMessageLog','numStatUpdateLog','numFollowersLog','numPeopleFollowingLog','utcOffset', 'creatTimestamp_year','hasUrl','category','verifStatus','textClass','pageClass','themeClass','isLocVisible','uLanguage','isViewSizeCustom']
        # else:
        #     self.pickFields_ = pickfields

    def cleanData(self,df):
        # self.dataset = dataset
        # basepath = util.getBasePath()
        # astr = "%s/data/%s.csv"
        # if dataset not in ['train','test']:
        #     print("Invalid dataset type, only train and test are supported")
        #     return ""
        # filename = astr % (basepath, dataset)
        # self.df = pd.read_csv(filename)
        # if dataset == 'test':
        #     self.df.columns = self.indexlist_[:-1]
        # else:
        #     self.df.columns = self.indexlist_
        self.df = df
        del df
        self.fillna()
        self.outlier_dealing()
        self.buildFeatures()
        # self.outlier_dealing()
        self.onehotencode() # encode discrete features
        self.rescaleData()  # take logarithmic of numerical features
        self.cut2box() # cut continuous data
        # self.processInf() # replace inf,-inf
        return self.df
        # if(dataset=='train'):
        #     return self.df[self.pickFields_]
        # else:
        #     return self.df[self.pickFields_[:-2]],self.df['id']

    def rescaleData(self):
        self.df['numFollowersLog'] = self.df['numFollowers'].map(lambda x: np.log10(1.5+x))
        self.df['numPeopleFollowingLog'] = self.df['numPeopleFollowing'].map(lambda x: np.log10(1.5+x))
        self.df['numStatUpdateLog'] = self.df['numStatUpdate'].map(lambda x: np.log10(1.5+x))
        self.df['numDMessageLog'] = self.df['numDMessage'].map(lambda x: np.log10(1.5+x))
        self.df['avgClickLog'] = self.df['avgClick'].map(lambda x: np.log10(1.5+x))
        # if(self.dataset=='train'):
        #     self.df['numPLikesLog'] = self.df['numPLikes'].map(lambda x: np.log10(1.5+x))

    def buildFeatures(self):
        # field Personal URL: 1 for not null, 0 for null

        '''frequency encode'''
        # # change utcOffset, category, uLanguage  to str
        # self.df['utcOffset'] = self.df['utcOffset'].apply(str)
        # self.df['category'] = self.df['category'].apply(str)
        # self.df['uLanguage'] = self.df['uLanguage'].apply(str)
        # # encode uLanguage, category，creatTimestamp_year, utcOffset
        # # print(self.df.isnull().sum())
        # encoder = ce.CountFrequencyCategoricalEncoder(encoding_method='frequency',
        #                                               variables=['uLanguage', 'category', 'creatTimestamp_year',
        #                                                          'utcOffset'])
        # # fit the encoder
        # encoder.fit(self.df)
        # # transform data
        # self.df = encoder.transform(self.df)

        # le = LabelEncoder()
        # le.fit(self.df['uLanguage'].unique())
        # self.df['uLanguage'] = le.transform(self.df['uLanguage'])
        #
        # le = LabelEncoder()
        # le.fit(self.df['category'].unique())
        # self.df['category'] = le.transform(self.df['category'])
        self.classifyColor()
        self.extractImg()
        self.extractLoc()
        self.df['hasUrl'] = self.df['url'].isnull()
        self.df['hasUrl'] = self.df['hasUrl'].map(str)
        # featu
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
        # labelcoder = LabelEncoder()
        # for column in category:
        #     self.df[column] = labelcoder.fit_transform(self.df[column])
        # category = self.df.loc[:,
        #            ['hasUrl', 'covImgStatus', 'verifStatus', 'isViewSizeCustom', 'isLocVisible', 'uLanguage','textClass','pageClass','themeClass']]
        # cat_num = category.shape[-1]
        # category = np.reshape(np.array(category), (-1, cat_num))
        # onehotcoder = OneHotEncoder()
        # labelfeatures = onehotcoder.fit_transform(category)
        category_df = pd.get_dummies(category, dummy_na=True)

        # pd.concat((self.df,pd.DataFrame('':labelfeatures)))
        # self.df['labelfeatures'] = labelfeatures
        self.df = pd.concat([self.df,category_df],axis=1,ignore_index=False)

    def extractLoc(self):
        # extract features from field "location"
        pass

    def extractImg(self):
        # extract features from profile image
        pass

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

        self.df['dftextr'] = dftextr
        self.df['dftextg'] = dftextg
        self.df['dftextb'] = dftextb
        self.df['dfpager'] = dfpager
        self.df['dfpageg'] = dfpageg
        self.df['dfpageb'] = dfpageb
        self.df['dfthemer'] = dfthemer
        self.df['dfthemeg'] = dfthemeg
        self.df['dfthemeb'] = dfthemeb
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
            qu_high = set.quantile(q=0.75)
            qu_low = set.quantile(q=0.25)
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
        Discretize numerical continuous data
        :return:
        '''
        pass

    def getPickIndex(self):
        '''
        Return the begining index for picking fields, the picked df will be df.iloc[:][self.pickIndStart:]
        :return:
        '''
        return self.pickIndStart
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
