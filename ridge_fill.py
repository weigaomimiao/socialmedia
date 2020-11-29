import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

df_train = pd.read_csv("./data/train.csv")
indexList = ['id','uname','url','covImgStatus','verifStatus','textColor','pageColor','themeColor','isViewSizeCustom','utcOffset','location','isLocVisible','uLanguage','creatTimestamp','uTimeZone','numFollowers','numPeopleFollowing','numStatUpdate','numDMessage','category','avgvisitPerSecond','avgClick','profileImg','numPLikes']
df_train.columns = indexList

data_set=df_train.loc[:,['numFollowers','numStatUpdate','numDMessage','avgvisitPerSecond','avgClick']]
test_set=data_set[data_set.isnull().T.any()]
test_x=test_set.loc[:,['numFollowers','numStatUpdate','numDMessage','avgClick']]


temp=data_set.append(test_set)
train_set=temp.drop_duplicates(keep=False)

train_x=train_set.loc[:,['numFollowers','numStatUpdate','numDMessage','avgClick']]
train_y=train_set['avgvisitPerSecond'].values


# test model with part of data
Xtrain,Xtest,Ytrain,Ytest=train_test_split(train_x,train_y,test_size=0.3,random_state=420)

# normalization
# normalize x
transformer = StandardScaler()
transformer.fit(train_x)
train_xnorm = transformer.transform(train_x,copy=True)
test_xnorm = transformer.transform(test_x,copy=True)
Xtrain_norm = transformer.transform(Xtrain,copy=True)
Xtest_norm = transformer.transform(Xtest,copy=True)

# normalize y
# ensure 2d array
train_y = np.expand_dims(train_y,axis=1)
Ytrain = np.expand_dims(Ytrain,axis=1)
Ytest = np.expand_dims(Ytest,axis=1)

transformery = StandardScaler()
transformery.fit(train_y)
train_ynorm = transformery.transform(train_y,copy=True)
Ytrain_norm = transformery.transform(Ytrain,copy=True)
Ytest_norm = transformery.transform(Ytest,copy=True)

model=Ridge(alpha=1,fit_intercept=True,normalize=True).fit(Xtrain_norm,Ytrain_norm)

model.fit(Xtrain_norm,Ytrain)
ytest_pred = transformery.inverse_transform(model.predict(Xtest_norm))
# ytest_pred = model.predict(Xtest_norm)
print(mean_squared_error(Ytest,ytest_pred))


# do prediction: fit (train_x, train_y) and then all test_x
# model.fit(train_xnorm,train_ynorm)
# print(model.coef_)
# print(transformery.inverse_transform(model.predict(test_xnorm)))