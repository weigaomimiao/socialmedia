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
import util
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from feature_engine import categorical_encoders as ce
from sklearn.feature_selection import SelectKBest, mutual_info_regression

class Loader():
    def __init__(self,pickfields=None):
        self.df = None

        self.indexlist_ = ['id','uname','url','covImgStatus','verifStatus','textColor','pageColor','themeColor','isViewSizeCustom','utcOffset','location','isLocVisible','uLanguage','creatTimestamp','uTimeZone','numFollowers','numPeopleFollowing','numStatUpdate','numDMessage','category','avgvisitPerSecond','avgClick','profileImg','numPLikes']
        self.stander = StandardScaler() # for standardization if needed
        if pickfields is None:
            self.pickFields_ = ['numPeopleFollowingLog','hasUrl','avgClickLog','category','verifStatus','numFollowersLog','textClass','pageClass','themeClass','numDMessageLog','numStatUpdateLog','isLocVisible','uLanguage','utcOffset', 'creatTimestamp_year','isViewSizeCustom','numPLikes','numPLikesLog']
        else:
            self.pickFields_ = pickfields

    def loadData(self,dataset):
        self.dataset = dataset
        basepath = util.getBasePath()
        astr = "%s/data/%s.csv"
        if dataset not in ['train','test']:
            print("Invalid dataset type, only train and test are supported")
            return ""
        filename = astr % (basepath, dataset)
        self.df = pd.read_csv(filename)
        if dataset == 'test':
            self.df.columns = self.indexlist_[:-1]
        else:
            self.df.columns = self.indexlist_

        self.fillna()
        self.recodeData() # encode discrete features
        self.rescaleData() # take logarithmic of numerical features
        # self.processInf() # replace inf,-inf

        if(dataset=='train'):
            return self.df[self.pickFields_]
        else:
            return self.df[self.pickFields_[:-2]],self.df['id']

    def rescaleData(self):
        self.df['numFollowers'].replace(0,0.5)
        self.df['numFollowersLog'] = self.df['numFollowers'].map(lambda x: np.log10(1+x))

        self.df['numPeopleFollowing'].replace(0,0.5) # in case log10(1)
        self.df['numPeopleFollowingLog'] = self.df['numPeopleFollowing'].map(lambda x: np.log10(1+x))
        # print(self.df['numPeopleFollowingLog'].isin([-np.inf]).sum())

        self.df['numStatUpdate'].replace(0,0.5)
        self.df['numStatUpdateLog'] = self.df['numStatUpdate'].map(lambda x: np.log10(1+x))



        self.df['numDMessage'].replace(0,0.5)
        # print((self.df['numPeopleFollowing'] < 0).sum())
        self.df['numDMessageLog'] = self.df['numDMessage'].map(lambda x: np.log10(1+x))
        # print(self.df['numDMessageLog'].isin([-np.inf]).sum())

        self.df['avgClick'].replace(0,0.5)
        self.df['avgClickLog'] = self.df['avgClick'].map(lambda x: np.log10(1+x))

        if(self.dataset=='train'):
            self.df['numPLikes'].replace(0,0.5)
            self.df['numPLikesLog'] = self.df['numPLikes'].map(lambda x: np.log10(1+x))

    def recodeData(self):
        # field Personal URL: 1 for not null, 0 for null
        # Then enconde hasurl
        self.df['hasUrl'] = self.df['url'].isnull()
        le = LabelEncoder()
        le.fit(self.df['hasUrl'].unique())
        self.df['hasUrl'] = le.transform(self.df['hasUrl'])

        # create_timpstamp_year
        self.df["creatTimestamp_year"] = self.df['creatTimestamp'].apply(lambda x: x.split()[-1]).tolist()
        # self.df["creatTimestamp_year"] = self.df['creatTimestamp_year'].apply(lambda x: int(x))
        # encode covImgStatus
        le.fit(self.df['covImgStatus'].unique())
        distr = le.classes_
        di = {distr[0]: 0, distr[1]: 1, distr[2]: 2}
        self.df.replace({"covImgStatus": di})

        # encode islocVisible
        le.fit(self.df['isLocVisible'].unique())
        self.isLocVisible_lowercase()
        self.df['isLocVisible'] = le.transform(self.df['isLocVisible'])

        # encode verifStstus
        le = LabelEncoder()
        le.fit(self.df['verifStatus'].unique())
        self.df['verifStatus'] = le.transform(self.df['verifStatus'])

        # encode isViewSizeCustom
        le = LabelEncoder()
        le.fit(self.df['isViewSizeCustom'].unique())
        self.df['isViewSizeCustom'] = le.transform(self.df['isViewSizeCustom'])

        '''frequency encode'''
        # change utcOffset, category, uLanguage  to str
        self.df['utcOffset'] = self.df['utcOffset'].apply(str)
        self.df['category'] = self.df['category'].apply(str)
        self.df['uLanguage'] = self.df['uLanguage'].apply(str)
        # encode uLanguage, category，creatTimestamp_year, utcOffset
        # print(self.df.isnull().sum())
        encoder = ce.CountFrequencyCategoricalEncoder(encoding_method='frequency',
                                                      variables=['uLanguage', 'category', 'creatTimestamp_year',
                                                                 'utcOffset'])
        # fit the encoder
        encoder.fit(self.df)
        # transform data
        self.df = encoder.transform(self.df)

        # le = LabelEncoder()
        # le.fit(self.df['uLanguage'].unique())
        # self.df['uLanguage'] = le.transform(self.df['uLanguage'])
        #
        # le = LabelEncoder()
        # le.fit(self.df['category'].unique())
        # self.df['category'] = le.transform(self.df['category'])

        self.extractImg()
        self.extractLoc()


    def extractLoc(self):
        # extract features from field "location"
        pass

    def extractImg(self):
        # extract features from profile image
        # y = self.df['numPLike']

        # self.df['newNameImg'] = "blabla"
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
        self.df['textClass'] = kmeans_text.labels_
        self.df['pageClass'] = kmeans_page.labels_
        self.df['themeClass'] = kmeans_theme.labels_

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

        #replace NaN in avgClick by predicton result
        filled_data = pd.concat([train_set, test_x], axis=0)
        filled_data = filled_data.sort_index()
        self.df['avgClick'] = filled_data['avgClick']


    def isLocVisible_lowercase(self):
        self.df['isLocVisible'] = self.df['isLocVisible'].str.lower()

    def fillna(self):
        # filling all blanks, can call like fillLoc etc.
        self.df['covImgStatus'].fillna('unknown', inplace=True)
        # fill location
        self.fillLoc()
        self.fillcategory()
        self.fillavgclick()
        self.classifyColor()

    # def processInf(self):
    #     self.df.replace(np.inf,999999)
    #     self.df.replace(-np.inf, 1e7)

    def fillLoc(self):
        # filling?
        pass

    def cutPNumLikes(self): # only do it for training set
        if(self.dataset=='test'):
            return
        bins = [0,10000,20000,np.max(self.df['pNumLikes'])]
        self.df['pNumLikesClass'] = pd.cut(self.df['pNumLikes'],bins)


class Standardize():
    def __init__(self):
        self.standrdX = StandardScaler()
        # self.standrdY = lambda x: np.power(10,x)

    def fit(self,X):
        self.standrdX.fit(X)
        # self.standrdY.fit(Y)

    def transform(self,X):
        # if Y is None:
        #     return self.standrdX.transform(X,copy=True)
        # else:
        self.standrdX.transform(X,copy=True)

    def inverse_log10_y(self,Y):
        ypred = np.power(10,Y)-1 # cause y is rescaled as log(1+x)
        ypred = np.ceil(ypred).astype(int)
        return ypred

class BuildDataset():
    def __init__(self,kbest=15,pickfields=None):
        self.standar = Standardize() # for standardization
        if(kbest>0):
            self.selector = FeaSelector(kbest) # for selecting features
        loader_train = Loader(pickfields)
        loader_test = Loader(pickfields)
        df_train = loader_train.loadData('train')

        # plot corr-heatmap
        util.matplot(df_train)
        # take values
        df_test,testId = loader_test.loadData('test')
        trainX, trainLogY = df_train.iloc[:, :-2].values, df_train['numPLikesLog'].values
        # test inf
        # tmp = np.isinf(trainX).sum(axis=0)

        trainY = df_train['numPLikes'].values
        testX= df_test.values

        # fitting selector and standar
        if(kbest>0):
            self.selector.fit(trainX,trainLogY)
        self.standar.fit(trainX)

        # train set: standardization and feature selection x
        self.trainX= self.standar.transform(trainX)
        if(kbest>0):
            self.trainX = self.selector.select(self.trainX)
        # test set: standardization and feature selection
        self.testX = self.standar.transform(testX)
        if(kbest>0):
            self.testX = self.selector.select(self.testX)

        # set y, expand dims for model
        trainLogY = np.expand_dims(trainLogY, axis=1)
        self.trainLogY = trainLogY
        self.trainY = np.expand_dims(trainY, axis=1)

        self.testId = testId

    def getData(self):
        return self.trainX,self.trainY,self.trainLogY,self.testX,self.testId,self.standar


class FeaSelector():
    def __init__(self,k):
        self.selector = SelectKBest(mutual_info_regression,k=k)

    def fit(self,X,y):
        self.selector.fit(X, y)
    def select(self,X):
        X_new = self.selector.transform(X)
        return X_new
