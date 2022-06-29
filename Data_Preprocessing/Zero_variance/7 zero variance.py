# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 11:11:11 2022

@author: rohit
"""

# import required libraries and set working directory path

import os
os.chdir((r'D:\360digitmg\lectures\3 Data preprocessing'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# load dataset 
df = pd.read_csv('Datasets/Z_dataset.csv')
df.head()

# check info
df.info() # 150x6, one int and object columns rest flaot data type, no null values.

# check 1st and 2nd moment business decisions
df.describe()
# std is nearly zero for every column. ID column is not usefull here.
# mean and median have little difference.

# remove ID column, not useful for further analysis.
df.drop(['Id'], axis=1, inplace=True) 
df.head()

# check missing value
df.isna().sum() # no null values
# check duplicates
df.duplicated().sum() # one found.
# remove duplicates
df1=df.drop_duplicates(keep='first')

# check outliers using box plot
sns.boxplot(df1['square.length']) # no outliers
sns.boxplot(df1['square.breadth']) # yes outliers
sns.boxplot(df1['rec.Length']) # no outliers
sns.boxplot(df1['rec.breadth']) # no outliers

# check values in colour variable
df1.colour.value_counts() # green, blue, orange.

# see count of outliers or manually checking outliers
IOR = df1['square.breadth'].quantile(0.75)-df1['square.breadth'].quantile(0.25)
upper_limit=df1['square.breadth'].quantile(0.75)+(IOR*1.5)
lower_limit=df1['square.breadth'].quantile(0.25)-(IOR*1.5)

# check outliers
outliers_df = np.where(df1['square.breadth']>upper_limit, True, np.where(df1['square.breadth']<lower_limit, True, False))
outliers_df.sum() # output=4, hence there are 4 outliers.

# replace outliers manuallay 
 df1['square.breadth']=pd.DataFrame(np.where(df['square.breadth']>upper_limit, upper_limit, np.where(df['square.breadth']<lower_limit, lower_limit, df['square.breadth'])))

# data is free from outliers, missing values, duplicats.
# check variance

df1['square.length'].var()
df1['square.breadth'].var()
df1['rec.Length'].var()
df1['rec.breadth'].var()

# except rec.length, all variance are nearly zero.
# check QQ  plot for normally distribution
from scipy import stats
import pylab
stats.probplot(df1['square.length'], dist='norm', plot=pylab) # normally distributed.
stats.probplot(df1['square.breadth'], dist='norm', plot=pylab) # normally distributed.
stats.probplot(df1['rec.Length'], dist='norm', plot=pylab) # not normally distributed.
stats.probplot(df1['rec.breadth'], dist='norm', plot=pylab) # not normally distributed.

# remove the object varible from data
num_df=df1.drop(['colour'], axis=1)

# use library to remove zero or near zero  variance ( near zero= < 0.6 variance)
from sklearn.feature_selection import VarianceThreshold
var_rem = VarianceThreshold(threshold=0.6)
var_rem.fit(num_df)

var_rem.get_support() # true = var > 0.6 and false = var <= 0.6

num_df.columns[var_rem.get_support()] # column more than 0.6 variance

# to remove near zero variance variable
NZero_var = [column for column in num_df.columns if column not in num_df.columns[var_rem.get_support()]]

print(NZero_var) # NEar zero variance

# no will drop near zero variance
num_df.drop(NZero_var, axis=1,inplace=True)

num_df.columns 

# check var
for column in num_df.columns:
    print(column)
    print(num_df[column].var())
    
# selected only var > 0.6 

######### end of the script