# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 20:50:16 2022

@author: rohit
"""


# Lasso & Ridge Regression Assignment on computer data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
data = pd.read_csv(r"D:\360digitmg\lectures\28 Lasso Ridge Regression\Assignment\Datasets\Life_expectencey_LR.csv")

data.columns
data = data.loc[:, ['Life_expectancy', 'Year','Adult_Mortality',
       'infant_deaths', 'Alcohol', 'percentage_expenditure', 'Hepatitis_B',
       'Measles', 'BMI', 'under_five_deaths', 'Polio', 'Total_expenditure',
       'Diphtheria', 'HIV_AIDS', 'GDP', 'Population', 'thinness',
       'thinness_yr', 'Income_composition', 'Schooling', 'Country','Status']]
data.info()
data.isna().sum()
cat_col = data.loc[:, ['Country','Status']]

# Impute Missing values
from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values = np.nan, strategy='mean')

data = pd.DataFrame(mean_imputer.fit_transform(data.drop(['Country', 'Status'], axis=1)))
data.columns = ['Life_expectancy', 'Year','Adult_Mortality',
       'infant_deaths', 'Alcohol', 'percentage_expenditure', 'Hepatitis_B',
       'Measles', 'BMI', 'under_five_deaths', 'Polio', 'Total_expenditure',
       'Diphtheria', 'HIV_AIDS', 'GDP', 'Population', 'thinness',
       'thinness_yr', 'Income_composition', 'Schooling']

data.isnull().sum() # No null 
data = pd.concat([data, cat_col], axis=1)
stats = data.describe()
data.duplicated().sum()

data.Country.value_counts()
data.Status.value_counts()
# drop country column
data.drop('Country', axis=1, inplace=True)

# one hot encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
data.Status = LE.fit_transform(data.Status)

# Scatter plot
sns.pairplot(data)
corr = data.corr()

# Preparing the model on train data 
f = ['+'.join(data.columns)]
model_train = smf.ols("Life_expectancy ~ Year+Adult_Mortality+infant_deaths+Alcohol+percentage_expenditure+Hepatitis_B+Measles+BMI+under_five_deaths+Polio+Total_expenditure+Diphtheria+HIV_AIDS+GDP+Population+thinness+thinness_yr+Income_composition+Schooling+Status", data = data).fit()
model_train.summary()

# Multicollinearity occurs when there are two or more independent variables in a multiple regression model,
# which have a high correlation among themselves. 

# prediction 
pred1 = model_train.predict(data)

# residual
res = data['Life_expectancy'] - pred1

# RMSE 
rmse1 = np.sqrt(np.mean(res**2))
rmse1 #  4.03
# rmse1 =  4.03 & Adj R^2 : 0.819


# Lasso Regression
from sklearn.linear_model import Lasso
# use GriedSearchCV to get best parameters
from sklearn.model_selection import GridSearchCV 

# create instance
lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lass_reg = GridSearchCV(lasso, parameters, scoring=('r2'), cv = 5)
lass_reg.fit(data.iloc[:, 1:], data.Life_expectancy)

lass_reg.best_params_
lass_reg.best_score_

lass_pred = lass_reg.predict(data.iloc[:, 1:])

# Adjusted r-square#
lass_reg.score(data.iloc[:, 1:], data.Life_expectancy)

# RMSE
np.sqrt(np.mean((lass_pred - data.Life_expectancy)**2))

# # rmse1 =  4.03 & Adj R^2 : 0.819
# rmse lasso = 4.03 & R^2 : 0.819

# Ridge Regression
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(data.iloc[:, 1:], data.Life_expectancy)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(data.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(data.iloc[:, 1:], data.Life_expectancy)

# RMSE
np.sqrt(np.mean((ridge_pred - data.Life_expectancy)**2))

# # rmse1 =  4.03 & Adj R^2 : 0.819
# rmse lasso = 4.03 & R^2 : 0.819
# rmse Ridge =  4.03 & R^2 : 0.819


# End of the script 