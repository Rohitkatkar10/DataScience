# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 20:40:53 2022

@author: rohit
"""

#  Multiple Linear Regression Assignment on computer data.

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

# loading data set
data = pd.read_csv(r"D:\360digitmg\lectures\24 Multiple Lin Regression\Assignment\Datasets_MLR\Computer_Data.csv")

data.columns
# price is label/tagert variable here.

data.info() 
data['Unnamed: 0'].value_counts() 
# Remove the column
data.drop(['Unnamed: 0'], axis=1, inplace=True)

# 1st and 2nd moment business decisions
stats = data.describe() # mean ~ median

# EDA
data.isna().sum() # no
data.duplicated().sum() # 76
data.drop_duplicates(keep='first', inplace=True)

# outliers
sns.boxplot(data.price) # yes
sns.boxplot(data.speed)
sns.boxplot(data.hd) # yes
sns.boxplot(data.ram) # yes
sns.boxplot(data.screen) # yes
sns.boxplot(data.ads)
sns.boxplot(data.trend)
# since it is possible that ram can be 32GB and screen size=17 inch. because this prices of laptop goes high.
# hence these outliers are not by chance.


# data visualization and comapre with target column

# speed
plt.figure(1) 
sns.histplot(data=data, x='speed') 
plt.figure(2)
sns.barplot(data=data, x='speed', y='price') 

# hd
plt.figure() 
sns.histplot(data=data, x='hd') 
plt.figure(2)
sns.barplot(data=data, x='hd', y='price') 

# ram 
plt.figure(1) 
sns.histplot(data=data, x='ram')
plt.figure(2)
sns.barplot(data=data, x='ram', y='price')  # no any relatoin

# screen
plt.figure(1) 
sns.histplot(data=data, x='screen')
plt.figure(2)
sns.barplot(data=data, x='screen', y='price')

# cd
plt.figure(1) 
sns.histplot(data=data, x='cd')
plt.figure(2)
sns.barplot(data=data, x='cd', y='price')

# multi
plt.figure(1) 
sns.histplot(data=data, x='multi')
plt.figure(2)
sns.barplot(data=data, x='multi', y='price')

# premium
plt.figure(1) 
sns.histplot(data=data, x='premium')
plt.figure(2)
sns.barplot(data=data, x='premium', y='price')

# ads
plt.figure(1) 
sns.histplot(data=data, x='ads')
plt.figure(2)
sns.barplot(data=data, x='ads', y='price')

# trend
plt.figure(1) 
sns.histplot(data=data, x='trend')
plt.figure(2)
sns.barplot(data=data, x='trend', y='price')

# See the relation among variables
sns.pairplot(data)
# from plot it is very difficult to see any pattern

# Get numeric value of correlation 
data.corr() # r = coefficient of correlation
# HD and ram have good +ve r value followed by speed and screen.

# get the dummy columns .
data_dum = pd.get_dummies(data, columns=['cd','multi','premium'], drop_first=False) 
corr = data_dum.corr()
# even though, a few columns have good correlation with price, but there are colinearity exists in data.
data_dum.info()
# we will use backward feature selection method to remove useless features.
# Model Training 
import statsmodels.formula.api as smf 

ml1 = smf.ols('price ~ speed + hd + ram + screen + ads + trend + cd_no + cd_yes + multi_no + multi_yes + premium_no + premium_yes', data=data_dum).fit()
ml1.summary()
# R^2 = 0.775 P-values > 0.05 for cd_no and multi_no. 

# drop these columns in dummy process.
data_dum1 = pd.get_dummies(data, columns=['cd','multi','premium'], drop_first=True) 
corr = data_dum1.corr()

# train on new summy data 
ml2 = smf.ols('price ~ speed + hd + ram + screen + ads + trend +  cd_yes +  multi_yes +  premium_yes', data=data_dum1).fit()
ml2.summary()
# R^2 = 0.775 , all p-value < 0.05 with same R^2 value.

# model training only on numeric data
ml3 = smf.ols('price ~ speed + hd + ram + screen + ads + trend', data=data).fit()
ml3.summary()
# R^2 = 0.711 , all p-value < 0.05 but here R^2 value descreased from 0.775.

# scale the data_dum1 then train
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
data_ss = pd.DataFrame(sc.fit_transform(data_dum1))
data_ss.columns = data_dum1.columns

data_ss.corr() # No change

# train scaled data
ml4 = smf.ols('price ~ speed + hd + ram + screen + ads + trend +  cd_yes +  multi_yes +  premium_yes', data=data_ss).fit()
ml4.summary()
# R^2 = 0.775 , all p-value < 0.05 with same R^2 value as for data_dum1.

# Not much change even after scaling the data. hence using ml2 model.
pred = ml2.predict(data_dum1)

# see normality of errors (it should be normally distributed)
from scipy import stats
res = ml2.resid 
stats.probplot(res, dist='norm', plot=plt)

# see pattern of error (there should be no pattern)
sns.residplot(x=pred, y=data_dum1.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()


# Now Train Final model
from sklearn.model_selection import train_test_split 
train, test = train_test_split(data_dum1, test_size=0.2) 

ml_final = smf.ols('price ~ speed + hd + ram + screen + ads + trend +  cd_yes +  multi_yes +  premium_yes', data=train).fit()
ml_final.summary()
# R^2 = 0.778

# prediction
pred_final = ml_final.predict(test)


# Test residual values
test_resid = test.price - pred_final

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse # 280.315

# train data predition
train_pred = ml_final.predict(train)

# train residual values
train_resid = train.price-train_pred 

# RMSE values for train data
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse #274.81

# RMSE is so high in both cases.

###### END of the Script ###

