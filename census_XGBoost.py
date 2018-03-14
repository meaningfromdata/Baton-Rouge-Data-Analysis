# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 17:47:29 2018

@author: David

XGBoost run on cen2010 dataframe

Regressing median house value for census tract against various demographic features from 2010 census data

Adapted from example with Boston Housing Dataset:
    http://www.blopig.com/blog/2017/07/using-random-forests-in-python-with-scikit-learn/
    
and these XGBoost tutorials/Kaggle submissions:

https://machinelearningmastery.com/xgboost-python-mini-course/    

https://www.kaggle.com/mburakergenc/predictions-with-xgboost-and-linear-regression

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

from xgboost import XGBRegressor



### read CSV file containing BR census data
cen2010 = pd.read_csv('C:\\Users\\David\\Documents\\Data Science Related\\Project_Bremer\\cen2010_reduced2010CensusData.csv')


### Check how many columns contain missing data
print(cen2010.isnull().any().sum(), ' / ', len(cen2010.columns))

### Check how many entries in total are missing 
print(cen2010.isnull().any(axis=1).sum(), ' / ', len(cen2010))


### drop any rows with missing data
cen2010_dropna = cen2010.dropna()

### select columns for features 
features = cen2010_dropna.iloc[: , :-1]

### select target variable
target = cen2010_dropna.iloc[: , -1]


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

### instantiate XGboost regressor
xgb = XGBRegressor(n_estimators=200, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=4)

### fit xgb regressor to training data 
xgb.fit(X_train, y_train)




predictions = xgb.predict(X_test)

print(explained_variance_score(predictions,y_test))





'''
### evaluate model fit on test data with several metrics 
predicted_train = rf.predict(X_train)
predicted_test = rf.predict(X_test)
test_score = r2_score(y_test, predicted_test)
spearman = spearmanr(y_test, predicted_test)
pearson = pearsonr(y_test, predicted_test)
print(f'Out-of-bag R-2 score estimate: {rf.oob_score_:>5.3}')
print(f'Test data R-2 score: {test_score:>5.3}')
print(f'Test data Spearman correlation: {spearman[0]:.3}')
print(f'Test data Pearson correlation: {pearson[0]:.3}')
'''


