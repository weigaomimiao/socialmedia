#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: main.py
@time: 2020/11/29 18:20
@desc: main function for running savedModel
'''
from cleaner import BuildDataset
from models import Model,XGBoost
from models import FineTuning
from util import saveResult, trainvalLossPlot
from sklearn.model_selection import learning_curve

def finetune(model,X,y):
    #grid search
    param_grid = {'max_depth': [4, 5, 6, 7, 8], 'learning_rate': [5e-2, 1e-3], 'n_estimators': [200, 300, 400, 500]}
    # param_grid = None
    fitter = FineTuning(model,cv=10,param_grid=param_grid,scoring='neg_mean_squared_log_error')
    model_best,params, best_score = fitter.finetuning(X,y)
    print('best_params')
    print(params)
    print('best_score: %.3f'%best_score)
    model_best.savemodel()
    return model_best,params

def predict(testId,testX,model,stander):
    '''
    Do prediction
    :param testId:
    :param testX:
    :param model: The chosen best model
    :return: None
    '''
    ypred = model.predict(testX)
    ypred = stander.inverse_log10_y(ypred)

    # save prediction to csv
    saveResult(testId, ypred)
    model.savemodel()


def trainVal(buildDict=None):
    ''''
    Those are supported: Pick one from each parameter.
    @kbest: 
        Integer. 
        -1, or any integer> 0 (when kbest>0, it means the final dataset for fitting model will have kbest features)
    @featuresEng:
        String 
        'num+onehot', 'box+onehot'
    @dropOutlierRatio: 
        float
        any one between (0,0.5) (0 and 0.5 are illegal)
    @discreteMethod:
        String
        None, 'interval','frequency','DT'. None for @featuresEng=='num_onehot', interval for pd.cut, frequency for pd.qcut, DT for Decision Tree
    @bins:
        Integer
    '''
    buildParams = {} # Note those are rules used for building dataset
    if buildDict==None:
        buildParams = {'kbest':-1,
                       'featuresEng':'num+onehot',
                       'dropOutlierRatio':0.25,
                       'discreteMethod': None,
                       'binsInterval':0.2}
    else:
        buildParams = buildDict

    builder = BuildDataset(kbest=buildParams['kbest'],
                           featuresEng=buildParams['featuresEng'],
                           dropOutlierRatio=buildParams['dropOutlierRatio'],
                           discreteMethod=buildParams['discreteMethod'])
    X, Y, Ylog, testX, testId, stander = builder.getData()

    # fit savedModel with part data and evaluation
    model = XGBoost()
    # from xgboost import XGBRegressor
    # model = XGBRegressor(n_jobs=-1,n_estimators=500,learning_rate=0.05,min_child_weight=5,max_depth=6,random_state=0)
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(model, X, Ylog, cv=15, n_jobs=-1,
                       train_sizes=[0.7, 0.8, 0.9, 1],
                       return_times=True, shuffle=True, scoring='neg_mean_squared_error')
    print('=='*10)
    print(buildParams)
    print('mean_train_scores, train with [70%,80%,90%,100%]', train_scores.mean(axis=1))
    print('mean_test_scores, train with [70%,80%,90%,100%]', test_scores.mean(axis=1))

if __name__ == "__main__":
    ########## For building features
    '''
    This region is for choosing the best way to build features. After running those
    below, you will get training scores and test scores. Do test it use different
    paramsDict and . To get details of what 'paramsDict' support, check the annotations 
    in trainVal().
    
    Example:
    >> buildDict = {'kbest':-1,
                       'featuresEng':'num+onehot',
                       'dropOutlierRatio':0.25,
                       'discreteMethod': None,
                       'bins':200}
    >> trainVal(buildDict=buildDict)  
    '''

    ########## For model params
    '''
    This region is for fine tuning model, after choosing the method of building features,
    use finetune(model,X,y)
    
    Example:
    
    >> model_best, params = finetune(model,X,y)
    '''
    ########## for prediction
    '''
    The model_best getting from finetune(model,X,y) will be used for prediction
    
    Example:
    >> predict(testId,testX,model_best,stander)
    '''
    ##########
    # training set and test set building, select kbest feature
    build_list = [
        {'kbest': -1,'featuresEng': 'num+onehot','dropOutlierRatio': 0.25,
         'discreteMethod': None,'bins': 200},
        {'kbest': -1, 'featuresEng': 'box+onehot', 'dropOutlierRatio': 0.25,
         'discreteMethod': 'DT', 'bins': 200},
        {'kbest': -1, 'featuresEng': 'box+onehot', 'dropOutlierRatio': 0.25,
         'discreteMethod': 'interval', 'bins': 200},
        {'kbest': -1, 'featuresEng': 'box+onehot', 'dropOutlierRatio': 0.25,
         'discreteMethod': 'frequency', 'bins': 200}
    ]
    # trainVal(build_list[0])
    # trainVal(build_list[1])
    # trainVal(build_list[2])
    # trainVal(build_list[3])

    #------------------- load data with best features building rules.
    builder = BuildDataset(kbest=-1, featuresEng='box+onehot',
                           dropOutlierRatio=0.3,discreteMethod="frequency")
    X, Y, Ylog, testX, testId, stander = builder.getData()

    # fit savedModel with part data and evaluation
    model = XGBoost()
    # from xgboost import XGBRegressor
    # model = XGBRegressor(n_jobs=-1,n_estimators=500,learning_rate=0.05,min_child_weight=5,max_depth=6,random_state=0)
    # train_sizes, train_scores, test_scores, fit_times, _ = \
    #     learning_curve(model, X, Ylog, cv=15, n_jobs=-1,
    #                    train_sizes=[0.7, 0.8, 0.9, 1],
    #                    return_times=True, shuffle=True, scoring='neg_mean_squared_error')
    # print('mean_train_scores, train with [70%,80%,90%,100%]', train_scores.mean(axis=1))
    # print('mean_test_scores, train with [70%,80%,90%,100%]', test_scores.mean(axis=1))

    #----------------- grid search
    # param_grid = {'max_depth': [4, 5, 6, 7, 8], 'learning_rate': [5e-2, 1e-3], 'n_estimators': [200, 300, 400, 500]}
    # # param_grid = None
    # fitter = FineTuning(model,cv=10,param_grid=param_grid,scoring='neg_mean_squared_log_error')
    # model_best,params, best_score = fitter.finetuning(X,Ylog)
    # print('best_params')
    # print(params)
    # print('best_score: %.3f'%best_score)
    # model_best.savemodel()

    # trainvalLossPlot(train_scores,test_scores,model_name)

    #----------------------predict
    model.fit(X,Ylog)
    predict(testId,testX,model,stander)




