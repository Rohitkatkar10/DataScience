# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 16:12:16 2022

@author: rohit
"""

# Assignment on Simple Linear Regression on Logistic data.

# import libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

churn = pd.read_csv(r"D:\360digitmg\lectures\23 Simple Linear Regression\Assignment\Datasets_SLR\emp_data.csv")
# As per question, Churn_out_rate is Target variable.
churn.columns


churn.info() 
churn.describe() # 1st & 2nd moment decisions, shape of the dataset values.
# std shows variation of data points from mean. 

# Graphical representation 
import matplotlib.pyplot as plt # mostly used for visualization purposes.

# bar plot 
plt.bar(height=churn.Churn_out_rate, x = np.arange(0,10,1)) 
# churn rate are in decreasing order.

plt.hist(churn.Churn_out_rate) # it is right skewed (somehow)

plt.boxplot(churn.Churn_out_rate) # No outliers.

plt.bar(height=churn.Salary_hike, x = np.arange(0,10,1)) 
# Increasing

plt.hist(churn.Salary_hike) # same as first variable right skewed and missing some values.


plt.boxplot(churn.Salary_hike) # right skewed and no putliers

# Scatter plot ( we can get correlation from this)
plt.scatter(x=churn['Salary_hike'], y=churn['Churn_out_rate'], color='r') #default blue color 
# we get 3 things from scatter plot
# 1.Direction (+ve or -ve): here -ve
# 2.Linearity:  overall shape is linear
# 3.Strength (weak, moderate, strong): we need r values for strength.
# we cannot define exact strength from graph, hence need r value.
# graphs are subjective in nature. ( subjective means based on opinion or feeling rather than on facts.)

# find correlation ( find r value)
np.corrcoef(x=churn['Salary_hike'], y=churn['Churn_out_rate']) # r = 0.818 we assume r>0.85 is strong but some auther says r>0.8 is strong.
# we will stick to |r|>0.85. this is called 'Thumb Rule'.
# hence strength is  moderate.
# correlation comes from covarience.
# r = -0.91 but |r| = 0.91 with strong relation.

# Covariance (Covariance indicates the relationship of two variables whenever one variable changes.)
# Numpy does not have a function to calculate the covarience between two variables directly.
# Function for calculating a covarience matrix is called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

np.cov(churn['Salary_hike'], churn['Churn_out_rate'])

# number is -ve hence direction is -ve.


# Model Building
# import library
# pip install statsmodels
import statsmodels.formula.api as smf 

# Simple Linear Regression
model1 = smf.ols('Churn_out_rate ~ Salary_hike', data = churn).fit() # ols= ordinary least square in this have y ~ x.
model1.summary() 
# R^2 = 0.831 and p-value <0.05

# Now predict 
pred1=model1.predict(pd.DataFrame(churn.Salary_hike))

# Regression line
plt.scatter(churn.Salary_hike, churn.Churn_out_rate) # gives only scatter data
plt.plot(churn.Salary_hike, pred1, 'r') # gives only line in red color, take feature as is and replace y ie. churn rate with pred1
plt.legend([ 'Observed data','Predicted line'])# gives names 
plt.show() # shows the data,  run all above line in one go.

# Error calculation 
res1=churn.Churn_out_rate-pred1  # Residue or error
res_sqr1=res1*res1    # taken square, since it will give Zero if we add all residue.
# np.mean(res1)    # just to prove without square, it give almost zero value.
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1

# record all the evaluation parameters
# 1. o/p:Churn_out_rate, i/p: Salary , r=-0.91, R^2=0.831, RMSE=3.99, Model= simple Linear Regression, p < 0.05
# To imporve model accuracy, we need increase r and R^2 values and descrease RMSE values (ideally we want error to be zero).


# Model building on Transformed data. 
# log transformation 
# y = Churn_out_rate and X=log(salary_hike)

plt.scatter(x=np.log(churn.Salary_hike), y=churn.Churn_out_rate, color='blue') # graph is almost same, lets check r value.
np.corrcoef(np.log(churn.Salary_hike), churn.Churn_out_rate) # r= -0.921, previously r=-0.91 , No much change in r value.

# model 
model2=smf.ols('Churn_out_rate ~ np.log(churn.Salary_hike)',data=churn).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(churn['Salary_hike']))

# Regression line
plt.scatter(np.log(churn.Salary_hike), churn.Churn_out_rate)
plt.plot(np.log(churn.Salary_hike),pred2, 'r')
plt.legend(['Observed Data','predicted line'])
plt.show()


# Error calculation 
res2=churn.Churn_out_rate-pred2  # Residue or error
res_sqr2=res2*res2    # taken square, since it will give Zero if we add all residue.
# np.mean(res2)    # just to prove without square, it give almost zero value.
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2


# record all the evaluation parameters
# 1. o/p:Churn_out_rate, i/p: Salary , r=-0.91, R^2=0.831, RMSE=3.99, Model= simple Linear Regression, p < 0.05
# after transformation wc replaced by log(wc).
# 2. o/p:Churn_out_rate, i/p: log(Salary) , r=-0.921, R^2=0.894, RMSE=3.78, Model= simple Linear Regression, p < 0.05


# take log(AT) and proceed
plt.scatter(x=(churn.Salary_hike), y=np.log(churn.Churn_out_rate), color='yellow') # graph is changed. scatter has reduced.
np.corrcoef(churn.Salary_hike, np.log(churn.Churn_out_rate)) # -0.93

# devlop the model
model3=smf.ols('np.log(Churn_out_rate) ~ Salary_hike', data=churn).fit()
model3.summary() # 0.874

pred3 = model3.predict(pd.DataFrame(churn['Salary_hike']))
pred3_at = np.exp(pred3) # taking anti-log 
pred3_at

# Regression line
plt.scatter((churn.Salary_hike), np.log(churn.Churn_out_rate))
plt.plot((churn.Salary_hike),pred3, 'r')
plt.legend(['Observed Data','predicted line'])
plt.show()

# Error calculation 
res3=churn.Churn_out_rate-pred3_at  # Residue or error
res_sqr3=res3*res3    # taken square, since it will give Zero if we add all residue.
# np.mean(res3)    # just to prove without square, it give almost zero value.
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3 # 3.54

# record all the evaluation parameters
# 1. o/p:Churn_out_rate, i/p: Salary , r=-0.91, R^2=0.831, RMSE=3.99, Model= simple Linear Regression, p < 0.05
# after transformation wc replaced by log(wc).
# 2. o/p:Churn_out_rate, i/p: log(Salary) , r=-0.921, R^2=0.894, RMSE=3.78, Model= simple Linear Regression, p < 0.05
#  no much change in values of r, R^2 increased and rmse deacresed, we will try other to reduce rmse
# we have taken log(y) ie log(AT)
# 3. o/p: log(chrun rate), i/p:(salary) , r=-0.93, R^2=0.874, RMSE=3.54, Model= simple Linear Regression (exponential model, since anti log of churn rate), p < 0.05  
# rmse decreasing and r increasing. R^2 deacresing but not by large margin.
# try another way to0 so we can increase R^2 too
# we need parabolic equation here. or polynomial eq. with degree 2.
# y = c + c1x + c2X^2. 

# this curvilinear nature comes when we took log(AT), we will keep this as is and will take square of Waist.

#### polynomial Transformation 
# x = Salary; x^2=Salary*Salary, y=log(churn rate)

model4=smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike)', data=churn).fit()
model4.summary() # 0.984

pred4 = model4.predict(pd.DataFrame(churn)) # values are in  log form
pred4_at = np.exp(pred4)  # anti-log values
pred4_at 

# regression (here we create polynomial features)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2) # selecting degree as 2 
X = churn.iloc[:,0:1].values # selecting column as Salary_hike with all rows
X_poly = poly_reg.fit_transform(X) # getting values of x^0, X^1 and x^2
# y = wcat.iloc[:,1].values


plt.scatter((churn.Salary_hike), np.log(churn.Churn_out_rate)) # plotting data
plt.plot(X,pred4, 'red')            # plotting prediction line
plt.legend(['Observed Data','predicted line'])
plt.show() 


# Error calculation 
res4=churn.Churn_out_rate-pred4_at  # Residue or error
res_sqr4=res4*res4    # taken square, since it will give Zero if we add all residue.
# np.mean(res4)    # just to prove without square, it give almost zero value.
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4 # 1.32


# record all the evaluation parameters
# 1. o/p:Churn_out_rate, i/p: Salary , r=-0.91, R^2=0.831, RMSE=3.99, Model= simple Linear Regression, p < 0.05
# after transformation wc replaced by log(wc).
# 2. o/p:Churn_out_rate, i/p: log(Salary) , r=-0.921, R^2=0.894, RMSE=3.78, Model= simple Linear Regression, p < 0.05
#  no much change in values of r, R^2 increased and rmse deacresed, we will try other to reduce rmse
# we have taken log(y) ie log(AT)
# 3. o/p: log(chrun rate), i/p:(salary) , r=-0.93, R^2=0.874, RMSE=3.54, Model= simple Linear Regression (exponential model, since anti log of churn rate), p < 0.05  
# rmse decreasing and r increasing. R^2 deacresing but not by large margin.
# try another way to0 so we can increase R^2 too
# we need parabolic equation here. or polynomial eq. with degree 2.
# y = c + c1x + c2X^2. 

# 4. o/p: log(churn rate), i/p:(salary & salary^2) , r= cannot find, R^2=0.984, RMSE=1.32, Model= simple Linear Regression (polynomial model), p < 0.05  


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR","Log model","Exp Model", "Poly model"]), "RMSE":pd.Series([rmse1,rmse2,rmse3,rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse


###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(churn, test_size=0.2)

plt.scatter(train.Salary_hike, np.log(train.Churn_out_rate))

plt.figure(2)
plt.scatter(test.Salary_hike, np.log(test.Churn_out_rate)) # lol

FinalModel = smf.ols('np.log(Churn_out_rate) ~ Salary_hike+I(Salary_hike*Salary_hike)', data=churn).fit()
FinalModel.summary()

# R^2 = 0.984, and all p values are less tha 0.05.

# predict on test data
test_pred = FinalModel.predict(pd.DataFrame(test))
pred_test_AT = np.exp(test_pred)
pred_test_AT


# model Evaluation on test data.
test_res = test.Churn_out_rate - pred_test_AT
test_sqrs = test_res*test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Now will compare these test results with train data set results.
# prediction on train data.
train_pred = FinalModel.predict(pd.DataFrame(train))
pred_train_AT = np.exp(train_pred)
pred_train_AT

# model evalution on train data.
train_res = train.Churn_out_rate - pred_train_AT
train_sqrs = train_res*train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

# test rmse = 1.72 and train rmse = 1.20 now much difference.

#### End of the script 

