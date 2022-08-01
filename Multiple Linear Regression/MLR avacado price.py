# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:15:19 2022

@author: rohit
"""

#  Multiple Linear Regression Assignment on avacado data.

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

# loading data set
data = pd.read_csv(r"D:\360digitmg\lectures\24 Multiple Lin Regression\Assignment\Datasets_MLR\Avacado_Price.csv")

data.columns
# Exploratory Data Analysis
# 1. Measure of Central Tendency
# 2. Measure of Dispertion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distribution of variables
# 6. Graphical representations ( histogram, box plot, bot plot, stem & leaf plot, bar plot, etc)

# columns Name : MPG= Miles per Gallon, HP=horse power, vol = size of the car, sp = top speed, wt = weight of car.

data.describe() # no null values

# Graphical Representation
import matplotlib.pyplot as plt # used for visualization

# Total_Volume
plt.bar(height=data['Total_Volume'], x = np.arange(0,18249,1))
plt.hist(data['Total_Volume']) 
plt.boxplot(data['Total_Volume']) 

# Joint Plot
sns.jointplot(x=data['Total_Volume'], y=data['AveragePrice']) 


# Count plot
plt.figure(1, figsize=(16, 10))
sns.countplot(data['Total_Volume'])
# count plot gives number of times same value is repeated on y-axis.

# Q-Q plot (to know whether data in normally distributed or not)
from scipy import stats 
import pylab 
stats.probplot(data['AveragePrice'], dist='norm', plot=pylab) # yes, Normally distributed.
plt.show()

# scatter plot between the variables along with histograms
sns.pairplot(data.iloc[:,:]) # SCatter plot for each variable.
# all variables are negatively related to averagePice. 
# pair plot are between 

# correlation matrix
cor = data.corr() # we are quantifying data, we need are 'r' value. 

#
# Preparing model considering all the variables
import statsmodels.formula.api as smf # for regression model
data.columns
ml1 = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags ++ Large_Bags  +  year', data=data).fit() #Regression model

# summary
ml1.summary()


## splitting the data into train and test data
from sklearn.model_selection import train_test_split 
train, test = train_test_split(data, test_size=0.2) 

# preparing the model on the train data
model_train = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags ++ Large_Bags  +  year', data=train).fit()

# prediction on test datasets
test_pred=model_train.predict(test)

# Test residual values
test_resid = test_pred - test.AveragePrice 

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse # 4.329 

# train data predition
train_pred = model_train.predict(train)

# train residual values
train_resid = train_pred - train.AveragePrice

# RMSE values for train data
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse #4.370



# ENd


