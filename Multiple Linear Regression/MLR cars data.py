# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:12:39 2022

@author: rohit
"""


#  Multiple Linear Regression Assignment on car data.

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

# loading data set
#data = pd.read_csv(r"D:\360digitmg\lectures\24 Multiple Lin Regression\Assignment\Datasets_MLR\ToyotaCorolla.csv") # doesn't work.

data=pd.read_csv(r"D:\360digitmg\lectures\24 Multiple Lin Regression\Assignment\Datasets_MLR\ToyotaCorolla.csv",encoding='unicode_escape')
data.shape # (1436, 38)
data.columns

# Out of 38 columns, we will use only these columns= price, age_08_04, KM, HP, cc, Doors, Gears, Quarterly_Tax, and Weight 
data = data.loc[:, ['Price', 'Age_08_04', 'KM', 'HP', 'cc', 'Doors', 'Gears', 'Quarterly_Tax', 'Weight']]

# price: Target variable, price of the car.
# age_08_04: Might be age in months.
# KM: distance covered by car til now. 
# HP: Engine power in Horse power (HP)
# cc: Engine power in cubic centimeter,
# Doors: Number of doors,
# Gears: Number of speed gears,
# Quarterly_Tax: tax, 
# Weight: weight of the car.

# 1st and 2nd moment business decisions
stats = data.describe() # mean ~ median

# EDA
data.isna().sum() # no
data.duplicated().sum() # 1
data.drop_duplicates(keep='first', inplace=True)

# outliers
sns.boxplot(data.Price)
sns.boxplot(data.Age_08_04)
sns.boxplot(data.KM)
sns.boxplot(data.HP) 
sns.boxplot(data.cc) 
sns.boxplot(data.Doors) # no
sns.boxplot(data.Gears)
sns.boxplot(data.Quarterly_Tax)
sns.boxplot(data.Weight)
# EXCEPT Doors, all have outliers.

# outliers Treatment
from feature_engine.outliers import Winsorizer
data.columns

winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails, 
                          fold = 1.5,
                          variables = ['Price', 'Age_08_04', 'KM', 'HP', 'cc', 'Gears',
                                 'Quarterly_Tax', 'Weight'])
                          
# now fit this model to outliers column.

data1 = winsor.fit_transform(data) # stored in new variable.

# Randomly confirming
sns.boxplot(data1.Age_08_04) # NO


# data visualization and comapre with target column

# Age_08_04
plt.figure(1) 
sns.histplot(data=data, x='Age_08_04') 
plt.figure(2)
sns.barplot(data=data, x='Age_08_04', y='Price') 

# KM 
plt.figure(1) 
sns.histplot(data=data, x='KM')
plt.figure(2)
sns.barplot(data=data, x='KM', y='Price')  # no any relatoin

# HP
plt.figure(1) 
sns.histplot(data=data, x='HP')
plt.figure(2)
sns.barplot(data=data, x='HP', y='Price')

# cc
plt.figure(1) 
sns.histplot(data=data, x='cc')
plt.figure(2)
sns.barplot(data=data, x='cc', y='Price')

# Doors
plt.figure(1) 
sns.histplot(data=data, x='Doors')
plt.figure(2)
sns.barplot(data=data, x='Doors', y='Price')

# Gears
plt.figure(1) 
sns.histplot(data=data, x='Gears')
plt.figure(2)
sns.barplot(data=data, x='Gears', y='Price')

# Quarterly_Tax
plt.figure(1) 
sns.histplot(data=data, x='Quarterly_Tax')
plt.figure(2)
sns.barplot(data=data, x='Quarterly_Tax', y='Price')

# Weight
plt.figure(1) 
sns.histplot(data=data, x='Weight')
plt.figure(2)
sns.barplot(data=data, x='Weight', y='Price')

# Scatter plot to see linearity, direction, strength.
sns.pairplot(data) 
plt.figure(2)
sns.pairplot(data1)


# Get numeric value of correlation 
corr = data.corr() # r = coefficient of correlation
corr1 = data1.corr()
data1.drop('Gears', axis=1, inplace=True)

# Q-Q plot (to know whether target in normally distributed or not)
from scipy import stats 
import pylab 
stats.probplot(data.Price, dist='norm', plot=pylab) # yes, Normally distributed.
plt.show()

plt.figure(2)
stats.probplot(data1.Price, dist='norm', plot=pylab) # yes, Normally distributed.
plt.show()
# Not are not Normally distributed.

# Preparing model considering all the variables
import statsmodels.formula.api as smf # for regression model
data.columns
ml1 = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Quarterly_Tax + Weight + Gears ', data=data).fit() #Regression model
ml1.summary()
# R^2 = 0.862, cc and doors have p-value > 0.05.

# remove Doors and cc column
ml2 = smf.ols('Price ~ Age_08_04 + KM + HP + Gears + Quarterly_Tax + Weight', data=data).fit() #Regression model
ml2.summary()

# use outlier treated data1
ml3 = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Quarterly_Tax + Weight', data=data1).fit() #Regression model
ml3.summary()
# not improving much R^2 value.

# use ml2 as final model
# prediction 
pred1 = ml2.predict(data)

# Q-Q plot to check whether errors are normally distributed.
res = ml2.resid

# Q-Q plot 
from scipy import stats
stats.probplot(res, dist='norm',plot=plt)
plt.show() # yes Normally distributed (we need normally distributed)

# residual is error
# residuals vs fitted plot
sns.residplot(x=pred1,y=data.Price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show() # No any pattern (we need no output)
# expected that errors should not have any pattern in them.
# No pattern in above plot, errors are independent of each other.


# Now Train Final model
from sklearn.model_selection import train_test_split 
train, test = train_test_split(data, test_size=0.2) 

ml = smf.ols('Price ~ Age_08_04 + KM + HP + Gears + Quarterly_Tax + Weight', data=train).fit() #Regression model
ml.summary()

# prediction
pred_final = ml.predict(test)


# Test residual values
test_resid = test.Price - pred_final

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse # 1327.60

# train data predition
train_pred = ml.predict(train)

# train residual values
train_resid = train.Price-train_pred 

# RMSE values for train data
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse #1343.323

# RMSE is so high in both cases.

###### END of the Script ###

