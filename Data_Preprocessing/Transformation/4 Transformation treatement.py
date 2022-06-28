# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:22:21 2022

@author: rohit
"""

# Transformation Assignment 

# import Necessary libraries.
import os
os.chdir(r'D:\360digitmg\lectures\3 Data preprocessing')

import pandas as pd # for data manupulation
data = pd.read_csv('Datasets/calories_consumed.csv')

# see data.
data.head()
data.info() # 14x2, no null values, all integer type data.
# data.shape # for data dimension. 
data.columns
data.dtypes # to get only type of data in columns.

# get stats
data.describe() # will give max, min, median, std, count values in one go.
# here we compare Median(50%) and mean values for outliers presence. if there are equal or nearly equal then No influence of outliers. else outliers are there.

# see unique values.
data['Weight gained (grams)'].unique() # all integer values.
data['Weight gained (grams)'].value_counts()  # both value_counts and unique give same output. 

data['Calories Consumed'].unique() # in both columns are numerics values are there and no abnormal value seen.

# find the null values, .isnull() is alias of .isna(). both are used for finding missing values.
data.isna().sum() # na = not available , NaN= not a number
data.isnull().sum()  # No null values here.

# check duplicates
data.duplicated().sum() # no duplicates.

# check outliers using boxplot.
# import seaborn or matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.boxplot(data['Weight gained (grams)'])   # using matplotlib library (here only wire box time visuals)
sns.boxplot(data['Weight gained (grams)'])   # using seaborn (good visuals)

plt.boxplot(data['Calories Consumed'])  #no ouliers in both the columns.

# check the varience of data.
data.var() # only numeric variables will give output. both columns have non-zero varience, this means these columns have more information.


# check for Data normally distributed.
# will use Q-Q plot. import libraries for the plot.

import scipy.stats as stats
import pylab


data.(Weight gained (grams)) # doesn't work. use data['colName'] when column name contains space seperator.

# for wieght gained (grams)
stats.probplot(data['Weight gained (grams)'], dist='norm', plot=pylab)
# Data to be normally distributed, data points plotted in plot must lie on the straight line.
# In this, case data point are away from straight line. hence data is not normal. 

# check for Colories Consumed column
stats.probplot(data['Calories Consumed'], dist='norm', plot=pylab)
# in this case, data is along with the straigth line. 


# column 1 is not normally distributed, need to make Normally distributed.
# try taking square of  the variable
stats.probplot(data['Weight gained (grams)']*data['Weight gained (grams)'], dist='norm', plot=pylab)
# taking cube
stats.probplot(data['Weight gained (grams)']*data['Weight gained (grams)']*data['Weight gained (grams)'], dist='norm', plot=pylab)
# magnifying is making data to go away from straight line.

# will reduce the values of variable.
# take square root of variable.
import numpy as np
stats.probplot(np.sqrt(data['Weight gained (grams)']), dist='norm', plot=pylab) # data going near to line.
# take cube root. 
stats.probplot(np.cbrt(data['Weight gained (grams)']), dist='norm', plot=pylab) # no change in sqrt position. 
# fourth root. 
stats.probplot(np.sqrt(np.sqrt(data['Weight gained (grams)'])), dist='norm', plot=pylab) 

# from all above,  cube root and fourth root not making any progress. data point are there like in sqrt plot.

# try reciprocal of variable
stats.probplot((1/(data['Weight gained (grams)'])), dist='norm', plot=pylab)  # this taking data point near to line 

# try log
stats.probplot(np.log(data['Weight gained (grams)']), dist='norm', plot=pylab)

# variable  'calories consumed in noramally distributed but weight gained (grams) not
# to make it Normally distributed we applied various techniques out of it, reciprocal makes 1st variable 
# normally dicributed. 


# adding reciprocal data column to dataframe 
data['reciprocal_weight_gained']=(1/data['Weight gained (grams)'])

# saving data 
data.to_csv('Normilised_colories_consumed.csv', encoding='utf-8')
os.getcwd() # get the path

#  ########## End of The Script. ###########


