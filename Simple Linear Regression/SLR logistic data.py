# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 12:13:28 2022

@author: rohit
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 19:54:38 2022

@author: rohit
"""

# Assignment on Simple Linear Regression on Logistic data.

# import libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cal = pd.read_csv(r"D:\360digitmg\lectures\23 Simple Linear Regression\Assignment\Datasets_SLR\delivery_time.csv")
# As per question, calories consumed is Target variable.
cal.columns
cal.columns = 'delivery_time', 'sorting_time' # for easy typing

# EDA
cal.info() # delivery time is target here.
# First and second moment business decisions
stats = cal.describe()

cal.isna().sum() # no null
cal.duplicated().sum() # no duplicates

# unique values
for col in cal.columns:
    print(col, '=', cal[col].nunique(), '\n', cal[col].unique())
    print(" ")
    # No any special character 
    
# check outliers
sns.boxplot(y=cal['delivery_time']) # NO 
plt.figure(2)  # to get new new empty plot
sns.boxplot(x=cal['sorting_time']) # NO 
# data lookds normally distributed

# Data visualization
sns.barplot(x='sorting_time', y='delivery_time', data=cal)
# if delivery time increases with increase in sortin time.


sns.histplot(x=cal['sorting_time'])
plt.figure(2)
sns.histplot(x=cal['delivery_time']) # Right skewed

# scatter plot
sns.scatterplot(x='sorting_time', y='delivery_time', data=cal)
# +ve direction, looks some linear, strength (need numeric value)
# Correlation 
np.corrcoef(x=cal['sorting_time'], y=cal['delivery_time'])
# if |r| > 0.85 then strength is strong or 0.5 to <=0.85 id mpderate. here r=0.82 hence moderate relation.


# covariance
np.cov(cal['delivery_time'],cal['sorting_time'] )


# Model building 
import statsmodels.formula.api as smf 

# simple linear regression
model1 = smf.ols('delivery_time ~ sorting_time', data=cal).fit()
model1.summary()
# coefficient of determination (R^2) = 0.682 and p-value = <0.05

# prediction
pred1 = model1.predict(pd.DataFrame(cal.sorting_time))

# Regression line
plt.scatter(x=cal.sorting_time, y=cal.delivery_time) # original data point spread 
plt.plot(cal.sorting_time, pred1, 'b') # (x, y, colour) just prediton line
plt.legend(['Observed Data', 'Predicted line'])
plt.show() 
# to get spread of data and predicton line.

# Error calculation
res1 = cal.delivery_time - pred1
res_sqr1 = res1**2
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
print(rmse1) # 2.79
 
# Recording all evaluation variables
# 1. model: simple linear, r=0.82, (R^2) = 0.682 rmse1=2.79

# Try to increase the model R^2 value and reduce rmse to zero
# taking log of TArget.

# take scatter plot To get relation data
plt.scatter(cal.sorting_time, np.log(cal.delivery_time))
np.corrcoef(cal.sorting_time, np.log(cal.delivery_time)) # 0.84

# taking feature log just to see r value
np.corrcoef(np.log(cal.sorting_time),(cal.delivery_time)) # 0.83

# model building
model2 = smf.ols('np.log(delivery_time) ~ sorting_time', data=cal).fit()
model2.summary() # R^2 = 0.711

pred2 = model2.predict(pd.DataFrame(cal.sorting_time)) # this is log values
pred2_exp = np.exp(pred2) # anti-log


# Regression line
plt.scatter(x=cal.sorting_time, y=np.log(cal.delivery_time)) # original data point spread 
plt.plot(cal.sorting_time, pred2, 'b') # (x, y, colour) just prediton line
plt.legend(['Observed Data', 'Predicted line'])
plt.show() 
# to get spread of data and predicton line.

# Error calculation
res2 = cal.delivery_time - pred2_exp
res_sqr2 = res2**2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
print(rmse2) # 2.94 previous = 2.79  error increasing.


# Recording all evaluation variables
# 1. model: simple linear, r=0.82, (R^2) = 0.682 rmse1=2.79
# 2. EXP model, r=0.84, (R^2) = 0.711 rmse1=2.92

# take log of sorting time i.e. feature
# take scatter plot To get relation data
plt.scatter(np.log(cal.sorting_time), (cal.delivery_time))
np.corrcoef(np.log(cal.sorting_time),(cal.delivery_time)) # 0.83

# model building
model3 = smf.ols('(delivery_time) ~ np.log(sorting_time)', data=cal).fit()
model3.summary() 

pred3 = model3.predict(pd.DataFrame(cal.sorting_time)) # this is log values



# Regression line
plt.scatter(x=np.log(cal.sorting_time), y=cal.delivery_time) # original data point spread 
plt.plot(np.log(cal.sorting_time), pred3, 'b') # (x, y, colour) just prediton line
plt.legend(['Observed Data', 'Predicted line'])
plt.show() 
# to get spread of data and predicton line.

# Error calculation
res3 = cal.delivery_time - pred3
res_sqr3 = res3**2
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
print(rmse3) # 2.73 previous = 2.79,  2.94  error decreasing.


# Recording all evaluation variables
# 1. model: simple linear, r=0.82, (R^2) = 0.682 rmse1=2.79
# 2. EXP model, r=0.84, (R^2) = 0.711 rmse1=2.94
# 3. log model, r=0.83, (R^2) = 0.695 rmse1=2.73

# from above all, log model gives best results. 

# hence traing model using Log model

# selecting simple linear model as best model'
from sklearn.model_selection import train_test_split 
train, test = train_test_split(cal, test_size=0.2)

# train model on train set using simple linear regression. 
model1 = smf.ols('delivery_time ~ np.log(sorting_time)', data=train).fit()
model1.summary() # 0.755


# prediction
pred1 = model1.predict(pd.DataFrame(test.sorting_time))

# Error calculation
res1 = test.delivery_time - pred1
res_sqr1 = res1**2
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
print(rmse1) # 3.23

# R^2 = 75.5% and rmse=3.23

print(pd.DataFrame({'Actula':test.delivery_time,'Prediction': pred1}))


### END of the script##########