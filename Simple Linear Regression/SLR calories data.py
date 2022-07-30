# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 19:54:38 2022

@author: rohit
"""

# Assignment on Simple Linear Regression on Calories data.

# import libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cal = pd.read_csv(r"D:\360digitmg\lectures\23 Simple Linear Regression\Assignment\Datasets_SLR\calories_consumed.csv")
# As per question, calories consumed is Target variable.

# EDA
cal.info()
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
sns.boxplot(y=cal['Weight gained (grams)']) # NO 
plt.figure(2)  # to get new new empty plot
sns.boxplot(x=cal['Calories Consumed']) # NO 

# Data visualization
sns.barplot(x='Weight gained (grams)', y='Calories Consumed', data=cal)
# as colorie consumption increases weight increases.

sns.histplot(x=cal['Calories Consumed'], bins=10) # little Right skewed
plt.figure(2)
sns.histplot(x=cal['Weight gained (grams)'], bins=10) # Right skewed

# scatter plot
sns.scatterplot(x='Weight gained (grams)', y='Calories Consumed', data=cal)
# +ve direction, looks some linear, strength (need numeric value)
# Correlation 
np.corrcoef(x=cal['Weight gained (grams)'], y=cal['Calories Consumed'])
# if |r| > 0.85 then strength is strong. here r=0.94 hence strong relation.

# covariance
np.cov(cal['Calories Consumed'],cal['Weight gained (grams)'] )


# model building
import statsmodels.formula.api as smf

# simple linear Regression (ordinary least square)
model1 = smf.ols('Calories Consumed ~ Weight gained (grams)', data=cal).fit()
cal.columns = 'weight', 'calories'
model1 = smf.ols('calories ~ weight', data=cal).fit()
model1.summary()

# R^2=0.897, if R^2>0.80 then strong correlation. more the R^2 value better it is.
# p values are less than 0.05

# predict
pred1=model1.predict(pd.DataFrame(cal.weight))

# Regression line
plt.scatter(cal.weight, cal.calories) # gives only scatter data
plt.plot(cal.weight, pred1, 'r') # gives only line in red color
plt.legend([ 'Observed data','Predicted line'])# gives names 
plt.show() # shows the data,  run all above line in one go.

# much distance in data points and regression line.

# Error calculation
res1 = cal.calories - pred1
res_sqr1 = res1**2 # taking square
mse1= np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
print(rmse1) #rmse should be zero or near to zero. 

# model1 : r = 0.94, R^2=0.897, RMSE1=232.83


# check Normality of data since error is more.
from scipy import stats
cal.columns
plt.figure(1)
stats.probplot(cal.weight, dist='norm', plot = plt) # Not along with line.
plt.title('wieght normality') 
plt.show()
# plt=pyplot
plt.figure(2)
stats.probplot(cal.calories, dist='norm', plot = plt) # Not along with line. 
plt.title('calories normality') 
plt.show()
# From graphs, only weight is not normally distributed.

# Make weight normally distributed
# take log(weight)
wt_log = np.log(cal.weight)
plt.figure(3)
stats.probplot(wt, dist='norm', plot = plt) # Not along with line.
plt.title('log wieght normality') 
plt.show()

# take sqr(weight)
wt2 = (cal.weight)**2
plt.figure(4)
stats.probplot(wt2, dist='norm', plot = plt) # Not along with line.
plt.title('sqr weight normality') 
plt.show()

# take sqrt(weight)
wt3 = np.sqrt(cal.weight)
plt.figure(5)
stats.probplot(wt3, dist='norm', plot = plt) # Not along with line.
plt.title('sqrt weight normality') 
plt.show()

# among all log(weight) gives better normality.

# Model building on Transformed data. 
# log transformation 
# y = calories and X=log(weight)

plt.scatter(x=wt_log, y=cal.calories, color='blue') # graph is almost same, lets check r value.
np.corrcoef(wt_log, cal.calories) # r=0.93, previously r=0.94 , change in r value.

# define anothe model
model2 = smf.ols('calories ~ np.log(weight)', data=cal).fit()
model2.summary()

pred2=model2.predict(pd.DataFrame(cal.weight))

# Regression line
plt.scatter(np.log(cal.weight), cal.calories)
plt.plot(np.log(cal.weight),pred2, 'r')
plt.legend(['Observed Data','predicted line'])
plt.show() # looks okay.

# Error calculation
res2 = cal.calories - pred2
res_sqr2 = res2**2 # taking square
mse2= np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
print(rmse2) #rmse should be zero or near to zero. 

# record all the evaluation parameters
# 1. o/p: calories, i/p: weight , r=0.94, R^2=0.897, RMSE=232.83, Model= simple Linear Regression, p < 0.05
# after transformation wc replaced by log(weight).
# 2. o/p: calories, i/p: log(weight) , r=0.93, R^2=0.878, RMSE=253.55, Model= simple Linear Regression (log model), p < 0.05
# since little change in values of r, R^2 decreased and rmse increased, we will change the strategy. since rmse supposed to be reduced.


# take log(colories)
plt.scatter(x=cal.weight, y=np.log(cal.calories), color='blue') # graph is not same, lets check r value.
np.corrcoef(cal.weight, np.log(cal.calories)) # r=0.89, previously r=0.94,r=0.93 , change in r value.

model3 = smf.ols('np.log(calories )~ weight', data=cal).fit()
model3.summary()

pred3=model3.predict(pd.DataFrame(cal.weight))
# take antilog of pred3
pred3 = np.exp(pred3)

# Regression line
plt.scatter(cal.weight, np.log(cal.calories))
plt.plot((cal.weight),pred3, 'r')
plt.legend(['Observed Data','predicted line'])
plt.show() # looks okay.

# Error calculation
res3 = cal.calories - pred3
res_sqr3 = res3**2 # taking square
mse3= np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
print(rmse3) #rmse should be zero or near to zero. 

# record all the evaluation parameters
# 1. o/p: calories, i/p: weight , r=0.94, R^2=0.897, RMSE=232.83, Model= simple Linear Regression, p < 0.05
# after transformation wc replaced by log(weight).
# 2. o/p: calories, i/p: log(weight) , r=0.93, R^2=0.878, RMSE=253.55, Model= simple Linear Regression (log model), p < 0.05
# since little change in values of r, R^2 decreased and rmse increased, we will change the strategy. since rmse supposed to be reduced.
# we have taken log(y) ie log(calories)
# 3. o/p: log(calories), i/p:(weight) , r=0.89, R^2=0.808, RMSE=272.42, Model= simple Linear Regression (exponential model, since anti log of colories), p < 0.05  
# rmse supposed to come down but increasing and r and R^2 are expected to increase, they are descreasing.
# this happens because, data (see scatter plot) is not along straight line but it is curvilinear form, straight line equation cannot fit here.
# we need parabolic equation here. or polynomial eq. with degree 2.
# y = c + c1x + c2X^2. 


# this curvilinear nature comes when we took log(AT), we will keep this as is and will take square of Waist.

#### polynomial Transformation 
# x = waist; x^2=waist*waist, y=log(AT)

model4=smf.ols('np.log(calories) ~ weight + I(weight*weight)', data=cal).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(cal)) # values are in  log form
pred4_at = np.exp(pred4)  # anti-log values
pred4_at 

# regression (here we create polynomial features)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2) # selecting degree as 2 
X = cal.iloc[:,0:1].values # selecting column as waist with all rows
X_poly = poly_reg.fit_transform(X) # getting values of x^0, X^1 and x^2
# y = wcat.iloc[:,1].values


plt.scatter((cal.weight), np.log(cal.calories)) # plotting data
plt.plot(X,pred4, 'red')            # plotting prediction line
plt.legend(['Observed Data','predicted line'])
plt.show() 


# Error calculation 
res4=cal.calories-pred4_at  # Residue or error
res_sqr4=res4*res4    # taken square, since it will give Zero if we add all residue.
# np.mean(res4)    # just to prove without square, it give almost zero value.
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4


# record all the evaluation parameters
# 1. o/p: calories, i/p: weight , r=0.94, R^2=0.897, RMSE=232.83, Model= simple Linear Regression, p < 0.05
# after transformation wc replaced by log(weight).
# 2. o/p: calories, i/p: log(weight) , r=0.93, R^2=0.878, RMSE=253.55, Model= simple Linear Regression (log model), p < 0.05
# since little change in values of r, R^2 decreased and rmse increased, we will change the strategy. since rmse supposed to be reduced.
# we have taken log(y) ie log(calories)
# 3. o/p: log(calories), i/p:(weight) , r=0.89, R^2=0.808, RMSE=272.42, Model= simple Linear Regression (exponential model, since anti log of colories), p < 0.05  
# rmse supposed to come down but increasing and r and R^2 are expected to increase, they are descreasing.
# this happens because, data (see scatter plot) is not along straight line but it is curvilinear form, straight line equation cannot fit here.
# we need parabolic equation here. or polynomial eq. with degree 2.
# y = c + c1x + c2X^2. 
# RMSE4 = 240, R^2 = 0.852 


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR","Log model","Exp Model", "Poly model"]), "RMSE":pd.Series([rmse1,rmse2,rmse3,rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse
# Here simple Linear model give best R^2, r and RMSE values results among all.

# Hence train model using simple linear Regression model
from sklearn.model_selection import train_test_split 
train, test = train_test_split(cal, test_size=0.2, random_state=0)

model1 = smf.ols('calories ~ weight', data=train).fit()
model1.summary()

# R^2=0.925, if R^2>0.80 then strong correlation. more the R^2 value better it is.
# p values are less than 0.05

# predict
pred1=model1.predict(pd.DataFrame(test.weight))

# Regression line
plt.scatter(cal.weight, cal.calories) # gives only scatter data
plt.plot(test.weight, pred1, 'r') # gives only line in red color
plt.legend([ 'Observed data','Predicted line'])# gives names 
plt.show() # shows the data,  run all above line in one go.

# much distance in data points and regression line.

# Error calculation
res1 = test.calories - pred1
res_sqr1 = res1**2 # taking square
mse1= np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
print(rmse1) #rmse should be zero or near to zero. 

print(pd.DataFrame({'Actula':test.calories,'Prediction': pred1}))


### END of the script##########