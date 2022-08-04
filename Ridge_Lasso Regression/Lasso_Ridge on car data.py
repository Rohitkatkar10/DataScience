# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 20:29:38 2022

@author: rohit
"""

# Lasso & Ridge Regression Assignment on computer data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
data=pd.read_csv(r"D:\360digitmg\lectures\24 Multiple Lin Regression\Assignment\Datasets_MLR\ToyotaCorolla.csv",encoding='unicode_escape')
data.shape # (1436, 38)
data.columns

# Out of 38 columns, we will use only these columns= price, age_08_04, KM, HP, cc, Doors, Gears, Quarterly_Tax, and Weight 
data = data.loc[:, ['Price', 'Age_08_04', 'KM', 'HP', 'cc', 'Doors', 'Gears', 'Quarterly_Tax', 'Weight']]

data.columns
data.info()
data.isna().sum()
data.duplicated().sum()
data.drop_duplicates(keep='first', inplace=True)

data.corr()

# Scatter plot
sns.pairplot(data)

# Preparing the model on train data 
f = ['+'.join(data.columns)]
model_train = smf.ols("Price ~ Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = data).fit()
model_train.summary()

# prediction 
pred1 = model_train.predict(data)

# residual
res = data['Price'] - pred1

# RMSE 
rmse1 = np.sqrt(np.mean(res**2))
rmse1 #  1337.59
# rmse1 =  1337.59 & Adj R^2 : 0.862


# Lasso Regression
from sklearn.linear_model import Lasso
# use GriedSearchCV to get best parameters
from sklearn.model_selection import GridSearchCV 

# create instance
lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lass_reg = GridSearchCV(lasso, parameters, scoring=('r2'), cv = 5)
lass_reg.fit(data.iloc[:, 1:], data.Price)

lass_reg.best_params_
lass_reg.best_score_

lass_pred = lass_reg.predict(data.iloc[:, 1:])

# Adjusted r-square#
lass_reg.score(data.iloc[:, 1:], data.Price)

# RMSE
np.sqrt(np.mean((lass_pred - data.Price)**2))

# rmse1 =  1337.59 & Adj R^2 : 0.862
# rmse lasso = 1342.00 & R^2 : 0.861

# Ridge Regression
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(data.iloc[:, 1:], data.Price)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(data.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(data.iloc[:, 1:], data.Price)

# RMSE
np.sqrt(np.mean((ridge_pred - data.Price)**2))

# rmse1 =  1337.59 & Adj R^2 : 0.862
# rmse lasso = 1342.00 & R^2 : 0.861
# rmse Ridge = 1337.98 & R^2 : 0.862


# End of the script 







