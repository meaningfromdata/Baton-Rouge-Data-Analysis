# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 17:47:29 2018

@author: David

Random Forest Regression run on cen2010 dataframe

Regressing median house value for census tract against various demographic features from 2010 census data

Adapted from example with Boston Housing Dataset:
    http://www.blopig.com/blog/2017/07/using-random-forests-in-python-with-scikit-learn/
    
Feature importance adapted from:
    http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import statsmodels.api as sm
plt.style.use('ggplot')

from sklearn.linear_model import LinearRegression
# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr




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


### instantiate LinearRegression object
lm = LinearRegression()


### fit multiple regression model to features and target
lm.fit(features, target)


### print some details on the model fit

print('Coefficients:', lm.coef_)

print('Intercept:', lm.intercept_)

print('R^2:', lm.score(features, target))



### feature names with coefficients (note that these coefficients estimated cannot be trusted without first assessing collinearity)
featuresWithCoef = pd.DataFrame(list(zip(features.columns, lm.coef_)), columns = ['features', 'coefficients'])


### evaluate model fit on test data with several metrics 
predicted = lm.predict(features)

test_score = r2_score(target, predicted)
spearman = spearmanr(target, predicted)
pearson = pearsonr(target, predicted)
print(f'Test data R-2 score: {test_score:>5.2}')
print(f'Test data Spearman correlation: {spearman[0]:.2}')
print(f'Test data Pearson correlation: {pearson[0]:.2}')





### generating a residual plot
plt.scatter(predicted, target - predicted)
plt.hlines(y = 0, xmin=0, xmax=600000)
plt.title('Multiple Regression Residuals: (True-Predicted) Value')
plt.ylabel('Residuals')

### plot distribution of residuals
sns.distplot(target - predicted)


### compute RMSE of best fit random forest regressor (result from grid search) in stepwise fashion
### SSE -> MSE -> RMSE
sse = np.sum((target - predicted)**2)
mse = sse/(features.shape[0])
rmse = mse**(1/2)






### instantiate another LinearRegression object to fit with standardized features
lm_scaled = LinearRegression()



### standardize features in test/train dataframe columns
scaler = StandardScaler().fit(features)
features_scaled = pd.DataFrame(scaler.transform(features), index=features.index.values, columns=features.columns.values)

# StandardScaler().fit(target)
# target_scaled = pd.DataFrame(scaler.transform(target.reshape(-1,1)), index=target.index.values, columns=target.columns.values)


### fit multiple regression model to scaled features and unscaled target
lm_scaled.fit(features_scaled, target)


print('Coefficients:', lm_scaled.coef_)

print('Intercept:', lm_scaled.intercept_)

print('R^2:', lm_scaled.score(features, target))



### feature names with coefficients (note that these coefficients estimated cannot be trusted without first assessing collinearity)
featuresScaledWithCoef = pd.DataFrame(list(zip(features_scaled.columns, lm_scaled.coef_)), columns = ['features', 'coefficients'])


### evaluate model fit on test data with several metrics 
predicted_scaled = lm_scaled.predict(features_scaled)

test_score = r2_score(target, predicted_scaled)
spearman = spearmanr(target, predicted_scaled)
pearson = pearsonr(target, predicted_scaled)
print(f'Test data R-2 score: {test_score:>5.2}')
print(f'Test data Spearman correlation: {spearman[0]:.2}')
print(f'Test data Pearson correlation: {pearson[0]:.2}')



