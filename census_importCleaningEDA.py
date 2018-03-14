# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 17:35:31 2018

@author: David
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 13:48:08 2018

@author: David

EDA of data for "Project Bremer" 

From Bremer's email on 

My interpretation of the problem is you want to create a neural network that can predict
1) Number of building permits per year
2) Total building cost per year
3) Property values per year
4) Delta of each value year over year

based on features
a) Latitude & Longitude
b) Demographics

Analysis of the model created should yield some insight into how neighborhoods are split up. Do they map to subdivisions? Do they map to demographic features?


Question:

Is change in parcel valuation improvement related to total building valuation within nearby area (1 mile?)


Property value
Change in property value over time 


Some of the code is this file adapted from:
    https://www.kaggle.com/mburakergenc/predictions-with-xgboost-and-linear-regression


"""

# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import statsmodels.api as sm
from scipy.stats import spearmanr, pearsonr



plt.style.use('ggplot')




### read CSV file containing BR census data
census_df = pd.read_csv('C:\\Users\\David\\Documents\\Data Science Related\\Project_Bremer\\Census_Demographics.csv')

### import CSV file that contains interpolated latitude and longitude for each census tract in Louisiana
# geoCoords_df = pd.read_csv('C:\\Users\\David\\Documents\\Data Science Related\\Project_Bremer\\2017_gaz_tracts_Louisiana.csv')

### put name for longitude as in CSV header on in pandas df (not sure why it didn't import)
# geoCoords_df = geoCoords_df.rename(columns={geoCoords_df.columns[7]: "INTPTLONG" })



### reduce the columns in geoCoords to GEOID and Latitude and Long prior to merging with census dataframes
# geoCoords_df = geoCoords_df[['GEOID','INTPTLAT','INTPTLONG']] 


### 'GEOID' in the geoCoords_df is the same as 'FIPS ID' in census_df so renaming to facilitate merging
# geoCoords_df.rename(columns={'GEOID':'FIPS ID'}, inplace=True)




### how many unique census tracts are there?
# census_df['TRACT'].nunique()



### split census dataframes into two dataframes: one containing year 2000 data and one containing year 2010 data 
census_2000_df = census_df[census_df['CENSUS YEAR']==2000]
census_2010_df = census_df[census_df['CENSUS YEAR']==2010]


### after split reset indices of new dataframes (drop=True prevents creation of new index column)
census_2000_df = census_2000_df.reset_index(drop=True)
census_2010_df = census_2010_df.reset_index(drop=True)




# census_2000_df['FIPS ID'].isin(list(geoCoords_df['GEOID'])) 
 

   
# census_2000_df['FIPS ID'].isin(census_2010_df['FIPS ID']) 


"""
census_2000_df = census_2000_df.merge(geoCoords_df, how='inner', on = 'FIPS ID')
census_2010_df = census_2010_df.merge(geoCoords_df, how='inner', on = 'FIPS ID')
"""

### reduce to most relevant columns in census_2010_df 
census_2010_trim_df = census_2010_df[['CENSUS YEAR', 'FIPS ID', 'TOTAL POPULATION', 
                                      'POPULATION WHITE', 'POPULATION BLACK', 'POPULATION ASIAN', 'POPULATION OTHER', 
                                      'MEDIAN AGE', 'MEDIAN HOUSEHOLD INCOME', 'HIGH SCHOOL MALE', 'HIGH SCHOOL FEMALE', 
                                      'ASSOCIATES DEGREE MALE', 'ASSOCIATES DEGREE FEMALE', 
                                      'BACHELORS DEGREE MALE', 'BACHELORS DEGREE FEMALE', 
                                      'MASTERS DEGREE MALE', 'MASTERS DEGREE FEMALE', 
                                      'PROFESSIONAL DEGREE MALE', 'PROFESSIONAL DEGREE FEMALE', 
                                      'DOCTORAL DEGREE MALE', 'DOCTORAL DEGREE FEMALE', 
                                      'PERCENT OWNER OCCUPIED', 'PERCENT RENTER OCCUPIED', 
                                      'VACANCY RATES', 'MEDIAN YEAR BUILT', 'MEDIAN HOUSE VALUE OWNER OCCUPIED']]


### get rid of dollar signs in "MEDIAN HOUSE VALUE OWNER OCCUPIED" 
census_2010_trim_df['MEDIAN HOUSE VALUE OWNER OCCUPIED'] = census_2010_trim_df['MEDIAN HOUSE VALUE OWNER OCCUPIED'].str.replace('$', '')
census_2010_trim_df['MEDIAN HOUSE VALUE OWNER OCCUPIED'] = census_2010_trim_df['MEDIAN HOUSE VALUE OWNER OCCUPIED'].astype(float)



### list of names of multiple columns I want to replace % sign in:
cols = ['PERCENT OWNER OCCUPIED', 'PERCENT RENTER OCCUPIED','VACANCY RATES']

### replace % sign with nothing (get rid of it) and then cast values as floats
census_2010_trim_df[cols] = census_2010_trim_df[cols].replace({'\%': ''}, regex=True)
census_2010_trim_df[cols] = census_2010_trim_df[cols].astype(float)



### round all values in census_2010_trim_df dataframe to have only two decimal places
# census_2010_trim_df = census_2010_trim_df.round(decimals = 2)


### summary stats for TOTAL POPULATION column
census_2010_trim_df['TOTAL POPULATION'].describe()

### last row clearly contains spurious data (e.g. 0 population) so will be dropped here
census_2010_trim_df = census_2010_trim_df.drop(census_2010_trim_df.index[-1])


### collect and compute (aggregate and normalize) columns of interest in new dataframe  
cen2010 = pd.DataFrame()

cen2010['white'] = (census_2010_trim_df['POPULATION WHITE']/census_2010_trim_df['TOTAL POPULATION'])*100
cen2010['black'] = (census_2010_trim_df['POPULATION BLACK']/census_2010_trim_df['TOTAL POPULATION'])*100
cen2010['asian'] = (census_2010_trim_df['POPULATION ASIAN']/census_2010_trim_df['TOTAL POPULATION'])*100
cen2010['other'] = (census_2010_trim_df['POPULATION OTHER']/census_2010_trim_df['TOTAL POPULATION'])*100


cen2010['hs'] = ((census_2010_trim_df['HIGH SCHOOL MALE'] + census_2010_trim_df['HIGH SCHOOL FEMALE'])/census_2010_trim_df['TOTAL POPULATION'])*100
cen2010['assoc'] = ((census_2010_trim_df['ASSOCIATES DEGREE MALE'] + census_2010_trim_df['ASSOCIATES DEGREE FEMALE'])/census_2010_trim_df['TOTAL POPULATION'])*100
cen2010['bach'] = ((census_2010_trim_df['BACHELORS DEGREE MALE'] + census_2010_trim_df['BACHELORS DEGREE FEMALE'])/census_2010_trim_df['TOTAL POPULATION'])*100
cen2010['post-bach'] = ((census_2010_trim_df['MASTERS DEGREE MALE'] + census_2010_trim_df['MASTERS DEGREE FEMALE'] + census_2010_trim_df['PROFESSIONAL DEGREE MALE'] + census_2010_trim_df['PROFESSIONAL DEGREE FEMALE'] + census_2010_trim_df['DOCTORAL DEGREE MALE'] + census_2010_trim_df['DOCTORAL DEGREE FEMALE'])/census_2010_trim_df['TOTAL POPULATION'])*100

cen2010['age'] = census_2010_trim_df['MEDIAN AGE']

cen2010['income'] = census_2010_trim_df['MEDIAN HOUSEHOLD INCOME']

cen2010['owner occ'] = census_2010_trim_df['PERCENT OWNER OCCUPIED']
cen2010['renter occ'] = census_2010_trim_df['PERCENT RENTER OCCUPIED']

cen2010['vacancy'] = census_2010_trim_df['VACANCY RATES']

cen2010['year built'] = census_2010_trim_df['MEDIAN YEAR BUILT']

cen2010['house value'] = census_2010_trim_df['MEDIAN HOUSE VALUE OWNER OCCUPIED']

### round all values in cen2010 dataframe to have only two decimal places
cen2010 = cen2010.round(decimals = 2)

### write dataframe into a csv file that doesn't include the index as a column in the csv file
# cen2010.to_csv('C:\\Users\\David\\Documents\\Data Science Related\\Project_Bremer\\cen2010_reduced2010CensusData.csv', index=False)



### grid of pair plots using basic matplotlib plotting functions
g = sns.PairGrid(cen2010.dropna(), vars=['income', 'year built','house value'])
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)


### grid of pair plots using seaborn plotting functions (note that upper and lower plots in grid can be specified separately)
# g = sns.PairGrid(cen2010.dropna(), vars=['income', 'year built','house value'])
# g = g.map_upper(sns.regplot)
# g = g.map_diag(sns.distplot)
# g = g.map_lower(sns.regplot)





### pairplot with subset of variables
# sns.pairplot(cen2010, vars=['white', 'house value'], dropna=True)



### individual scatterplots with regression line
# sns.regplot(x = 'white', y = 'house value', data = cen2010)

# sns.regplot(x = 'black', y = 'house value', data = cen2010)

# sns.regplot(x = 'asian', y = 'house value', data = cen2010)

# sns.regplot(x = 'hs', y = 'house value', data = cen2010)

# sns.regplot(x = 'year built', y = 'house value', data = cen2010)

# sns.regplot(x = 'owner occ', y = 'house value', data = cen2010)

# sns.regplot(x = 'vacancy', y = 'house value', data = cen2010)



### drop any rows with missing data
cen2010_dropna = cen2010.dropna()

### select columns for features 
features = cen2010_dropna.iloc[: , :-1]

### select target variable
target = cen2010_dropna.iloc[: , -1]


### create dictionary of pearson correlation of each feature vs target 
feature_names = features.columns.tolist()
corr_dict = {}
for f in feature_names:
    key = f + ' vs ' + target.name
    corr_dict[key] = pearsonr(features[f].values, target.values)[0]

### create dataframe from corr_dict and transpose it for ease of viewing
corr_df = pd.DataFrame(corr_dict, index=['Value']).transpose()



### compute and plot full correlation matrix of features
corrMat_df = features.corr()
sns.heatmap(corrMat_df)
plt.yticks(rotation=0) # orients the y-axis row labels horizontally
plt.show()





