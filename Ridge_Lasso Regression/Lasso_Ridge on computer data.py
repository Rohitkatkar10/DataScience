# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 17:20:31 2022

@author: rohit
"""

# Lasso & Ridge Regression Assignment on computer data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
data = pd.read_csv(r"D:\360digitmg\lectures\28 Lasso Ridge Regression\Assignment\Datasets\Computer_Data (1).csv")

data.columns
data.drop('Unnamed: 0', axis = 1, inplace=True)
data.info()
data.isna().sum()
data.duplicated().sum()
data.drop_duplicates(keep='first', inplace=True)

data.corr()
data.cd.value_counts()
data.multi.value_counts()
data.premium.value_counts()

# one hot encoding
data1 = pd.get_dummies(data, drop_first='first')
data1.corr()

sns.pairplot(data1)

# Preparing the model on train data 
f = ['+'.join(data1.columns)]
model_train = smf.ols("price ~ speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes", data = data1).fit()
model_train.summary()

# prediction 
pred1 = model_train.predict(data1)

# residual
res = data['price'] - pred1

# RMSE 
rmse1 = np.sqrt(np.mean(res**2))
rmse1 # 275.90
# rmse1 = 275.90 & Adj R^2 : 0.775


# Lasso Regression
from sklearn.linear_model import Lasso
# use GriedSearchCV to get best parameters
from sklearn.model_selection import GridSearchCV 

# create instance
lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lass_reg = GridSearchCV(lasso, parameters, scoring=('r2'), cv = 5)
lass_reg.fit(data1.iloc[:, 1:], data1.price)

lass_reg.best_params_
lass_reg.best_score_

lass_pred = lass_reg.predict(data1.iloc[:, 1:])

# Adjusted r-square#
lass_reg.score(data1.iloc[:, 1:], data1.price)

# RMSE
np.sqrt(np.mean((lass_pred - data1.price)**2))

# rmse1 = 275.90 & Adj R^2 : 0.775
# rmse lasso = 275.90 & R^2 : 0.775

# Ridge Regression
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(data1.iloc[:, 1:], data1.price)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(data1.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(data1.iloc[:, 1:], data1.price)

# RMSE
np.sqrt(np.mean((ridge_pred - data1.price)**2))

# rmse1 = 275.90 & Adj R^2 : 0.775
# rmse lasso = 275.90 & R^2 : 0.775
# rmse Ridge = 275.90 & R^2 : 0.775

# All are same 

# End of the script 







