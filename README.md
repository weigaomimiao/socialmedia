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