# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:55:22 2022

@author: rohit
"""


#  Multiple Linear Regression Assignment on startup data.

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

# loading data set
data = pd.read_csv(r"D:\360digitmg\lectures\24 Multiple Lin Regression\Assignment\Datasets_MLR\50_Startups.csv")

data.columns
# For easy typing
data.columns = ['RD_Spend', 'Administration', 'Marketing_Spend', 'State', 'Profit']
# profit is label/tagert variable here.
# R&D spend = Research and development spending
# administartion = Administration cost
# state = states where startup is finctioning
# profit = profit earned by startups

data.info() 
data.State.value_counts() # only three unique values.

# 1st and 2nd moment business decisions
stats = data.describe() # mean ~ median

# EDA
data.isna().sum() # no
data.duplicated().sum() # NO

# outliers
plt.figure(1)
sns.boxplot(data['RD_Spend'])
# plt.boxplot(data['RD_Spend'])
plt.figure(2)
sns.boxplot(data['Administration'])
plt.figure(3)
sns.boxplot(data['Marketing_Spend'])
plt.figure(4)
sns.boxplot(data['Profit'])
# None except profit, has outliers 

# data visualization and comapre with target column
# RD spend
plt.figure(1) 
sns.histplot(data=data, x='RD_Spend') # little right skew
plt.figure(2)
sns.barplot(data=data, x='RD_Spend', y='Profit') # directly proportional

# administration
plt.figure() 
sns.histplot(data=data, x='Administration') # looks normally distributes
plt.figure(2)
sns.barplot(data=data, x='Administration', y='Profit') # no any relatoin

# marketing spend
plt.figure(1) 
sns.histplot(data=data, x='Marketing_Spend')
plt.figure(2)
sns.barplot(data=data, x='Marketing_Spend', y='Profit')  # no any relatoin

# state
plt.figure(1) 
sns.histplot(data=data, x='State')
plt.figure(2)
sns.barplot(data=data, x='State', y='Profit')

# See the relation among variables
sns.pairplot(data)
# clearly, profit and R&D spend has good correlation follwed by profit-marketing_spend
# profit-admin has no relation. # then marketing - RD has some relation in +ve direction called coliniarity.

# Get numeric value of correlation 
data.corr() # r = coefficient of correlation

# since state column Not affecting profit much.
data_dum = pd.get_dummies(data, columns=['State'], drop_first=True) 
corr = data_dum.corr()
# state column is not much related to profit. hence ignore it.

# Model Training 
import statsmodels.formula.api as smf 

ml1 = smf.ols('Profit ~ RD_Spend + Administration + Marketing_Spend', data=data).fit()
ml1.summary()
# R^2 = 0.951 P-values > 0.05 for administration and marketing spend. 

# Checking whether data has any influencial values
# Influence index plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)

# Index 76 is showing high influence so we can exclude that entire row.
#RD is zero here, see what happens by removing it
data_new = data.drop(data.index[49])

ml_new = smf.ols('Profit ~ RD_Spend + Administration + Marketing_Spend', data=data_new).fit()
ml_new.summary()
# R^2 = 0.91 P-values > 0.05 for administration and marketing spend. although marketing p-value is reduce from 0.105 to  0.075.

# Methods to remove variable
# method2
# we have to remove one of the column from admin and marketing.
# VIF = variance Infletion Factor ( used to detect the sevearity in multicollinearity in OLS regression analysis)
# VIF must not > 10, if yes then variable is correlated.

rsq_RD = ml_new = smf.ols('RD_Spend ~ Administration + Marketing_Spend', data=data).fit().rsquared
vif_RD = 1/(1-rsq_RD)

rsq_ad = ml_new = smf.ols('Administration  ~ RD_Spend + Marketing_Spend', data=data).fit().rsquared
vif_ad = 1/(1-rsq_ad)

rsq_mar = ml_new = smf.ols('Marketing_Spend  ~ Administration + RD_Spend', data=data).fit().rsquared
vif_mar = 1/(1-rsq_mar)

# storing vif values in a data frame
d1 = {'Variables':['RD','Admin','mar'], 'VIF':[vif_RD,vif_ad,vif_mar]}

vif_frame = pd.DataFrame(d1)
vif_frame
# all values are less than 10. hence it is hard to remove variable.

# Method 2
# we will compare admin and marketing_spend with profit for their correlation, will remove variables whose |r| value is less.
# since p value for both admin and marketing_spend is >0.05 and their correlation is also high.
data.corr() # r = coefficient of correlation
# profit-admin: r = 0.2, hence will remove adminstration column.


# ML model without adminstration
ml2 = smf.ols('Profit ~ RD_Spend +  Marketing_Spend', data=data).fit()
ml2.summary()
# R^2 = 0.95, p-value is more than 0.05 for marketing spend.

data1 = data.drop(['Administration'], axis=1)

data1.corr()

# ML model without adminstration and Marketing_Spend
ml3 = smf.ols('Profit ~ RD_Spend ', data=data).fit()
ml3.summary()
# R^2 = 0.947, p-value is < 0.05

# Hence profit is merely dependent on RD spending.
# prediction
pred = ml3.predict(data)


# Q-Q plot to check whether errors are normally distributed.
# Q-Q plot method 1
res = ml3.resid
sm.qqplot(res)
plt.show()

# Q-Q plot method 2
from scipy import stats
stats.probplot(res, dist='norm',plot=plt)
plt.show()


# residual is error
# residuals vs fitted plot
sns.residplot(x=pred,y=data.Profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()
# expected that errors should not have any pattern in them.
# No pattern in above plot, errors are independent of each other.


# Now Train Final model
from sklearn.model_selection import train_test_split 
train, test = train_test_split(data, test_size=0.2) 

ml = smf.ols('Profit ~ RD_Spend ', data=train).fit()
ml.summary() # R^2=0.939

# prediction
pred_final = ml.predict(test)


# Test residual values
test_resid = test.Profit - pred_final

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse # 8575.92

# train data predition
train_pred = ml.predict(train)

# train residual values
train_resid = train.Profit-train_pred 

# RMSE values for train data
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse #9440.78

# RMSE is so high in both cases.

###### END of the Script ###

