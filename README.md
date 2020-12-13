# socialmedia
## [Course project](https://www.kaggle.com/c/ift6758-a20/overview)
__Authors: Wei Wei, Yinghan Gao, Milan Mao, Miao Feng__

### Task
The main task for the data challenge is to predict the number of 'likes' for given details about the simulated profiles of users on social media.

You have been provided with various attributes about the visual aspects of the users' social media profiles, the basic details, the profile pictures used in the simulation. With the is information, you need to predict how many 'likes' the users could've likely received.

There is a CSV file of various collected features that has been provided, in addition to images showing the profile pictures of the users (based on the simulation)

## Feature Engineering
* Average visit per second is not useful.
* Fit whole dataset rather than KFold.

## Notes of scores
* Dec. 5: 2.17
outlier (0.75), log(1.5+x), old fillavgClick, mlp
* Dec. 6: 1.75
new fillavgClick, xgboost (not fine tuned), dummy_one_hot, train and test together to be cleaned and normalized
* Dec. 9: 
1.792:interval cutting, interval_log=0.2
1.795: frequency cutting, interval_log=0.05
* Dec.12
1. 1.7558：ulanguage target encoding, xgboost
2. 1.7627: no outlier, ulanguage target encoding, xgboost
3.       : utcoffset target


* possible features:
    * message+visit
    
    
## some notes
amx_depth=8,min_child_weight=5
mean_train_scores, train with 100%,cv=10 [-0.0001783  -0.00660658 -0.02118415 -0.03793948 -0.05579909]
mean_test_scores, train with 100%,cv=10 [-0.66222096 -0.59896329 -0.58126889 -0.57005199 -0.56282338]
max_depth=5,max_depth=6
mean_train_scores, train with 100%,cv=10 [-0.00023259 -0.00685624 -0.0209876  -0.03818473 -0.05553998]
mean_test_scores, train with 100%,cv=10 [-0.64749621 -0.595664   -0.58487519 -0.58417402 -0.56186128]
min_child_weight=3,max_depth=5,
mean_train_scores, train with 100%,cv=10 [-0.00731991 -0.08780463 -0.15569561 -0.20136152 -0.24042308]
mean_test_scores, train with 100%,cv=10 [-0.64762934 -0.59722366 -0.56245079 -0.55399954 -0.54620049]
