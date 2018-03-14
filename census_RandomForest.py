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
# import seaborn as sns
# import statsmodels.api as sm

plt.style.use('ggplot')

# from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

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

### test/train split of data
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.80, random_state=1)

### standardize features in test/train dataframe columns (good to do prior to PCA)
# scaler = StandardScaler().fit(X_train)
# X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index.values, columns=X_train.columns.values)
# X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index.values, columns=X_test.columns.values)

### instantiate and fit RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=0)
rf.fit(X_train, y_train)

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


### look at feature importance in regression
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]



### Print the feature ranking
print("Feature ranking:")

for f in range(features.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



### Plot the feature importances of the random forest regressor
plt.figure()
plt.title("Feature importances")
plt.bar(range(features.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(features.shape[1]), indices)
plt.xlim([-1, features.shape[1]])
plt.show()






### Commented out the k-fold cross-validation because rows order is structured

### search for an optimal value of estimators for Random Forest Regressor
### using 10-fold cross-validation

'''
nestimator_range=range(100,1100,100)
n_scores=[]
for n in nestimator_range:
    rf = RandomForestRegressor(n_estimators=n)
    acc_score= cross_val_score(rf, features, target, cv=10)
    n_scores.append(acc_score.mean())
print(n_scores)
'''


### search for an optimal number of features for Random Forest Regressor
### using 10-fold cross-validation

'''
nfeatures_range=range(1,features.shape[1],1)
n_scores=[]
for n in nfeatures_range:
    rf = RandomForestRegressor(n_estimators=100, max_features=n)
    acc_score= cross_val_score(rf, features, target, cv=10)
    n_scores.append(acc_score.mean())
print(n_scores)
'''




### census tract order (row order is presumably related to geographic proximity) means 
### that simple k-fold cross validation is not appropriate.  
### To randomize rows using shuffle-split cross validation (repeated random draws of samples from data for test/train)

shuffles = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
rf = RandomForestRegressor(n_estimators=100, max_features=4)
cross_val_score(rf, features, target, cv=shuffles)



### search for an optimal values of hyperparameters for Random Forest Regressor
### using grid search with the shuffled samples as above 

nestimator_range= range(100,1000,100)
nfeatures_range = range(1, 6, 1)

gs = GridSearchCV(
        RandomForestRegressor(),
        param_grid = {'max_features': nfeatures_range, 'n_estimators': nestimator_range},
        cv=shuffles
        )

### apply the grid search to the features and target
gs.fit(features, target)

### print details on best fit result from grid search
print("best estimator : ", gs.best_estimator_)
print("best params : ", gs.best_params_)
print("r^2 : ", gs.best_score_)



### generating a residual plot

### gs.predict() method uses best-fit found in grid search
predicted = gs.predict(features)
plt.scatter(predicted, target - predicted)
plt.hlines(y = 0, xmin=0, xmax=600000)
plt.title('Random Forest Regression Residuals: (True-Predicted) Value')
plt.ylabel('Residuals')


### plot distribution of residuals
sns.distplot(target - predicted)



### compute RMSE of best fit random forest regressor (result from grid search) in stepwise fashion
### SSE -> MSE -> RMSE
sse = np.sum((target - predicted)**2)
mse = sse/(features.shape[0])
rmse = mse**(1/2)
