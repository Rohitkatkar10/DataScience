# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 18:51:36 2022

@author: rohit
"""
# set working directory
import os
os.chdir('D:/360digitmg/lectures/3 Data preprocessing/Datasets')

# load the data set
import pandas as pd
data=pd.read_csv('Claimants.csv')

# Data preprocessing
data.info() # count says there are null values. 
data.shape # 1340x7 dimension
data.columns


# get stats
stats = data.describe()


# check missing values
data.isnull().sum()   # many null values are there.

# check duplicates
data.duplicated().sum() # No duplicates

# check unique values in missing data features
data['CLMSEX'].unique() #binary data = array([ 0.,  1., nan])
# data['CLMSEX'].value_counts() # total = 1428, nan is not counted. 
data['CLMSEX'].nunique() # gives two unique values. Nan is not unique value here. 

data['CLMINSUR'].unique() # 1, 0 and Nan.
data['SEATBELT'].unique()  # 1, 0 and Nan.
data['CLMAGE'].unique() # Nan is there, rest many float values are there.

# get count for binary data
data['CLMSEX'].value_counts() # 1.0 = 742, 0.0 = 586 ; majority is 1.
data['CLMINSUR'].value_counts() # 1.0 = 1179, 0.0 = 120 ; majority is 1.
data['SEATBELT'].value_counts() #  0.0 = 1270, 1.0 = 22 ; majority is 1.


# Hence for binary data, will use mode (since missing values are not more (<50) for every binary data variable, it will not imbalance the data)
#  and for CLMAGE will use mean.
# since for CLMAGE we want to use mean, will use manual method using '.fillna()'.

# to check
x = data['CLMAGE'].fillna(data.CLMAGE.mean()) 
x.unique() # no null values seen

x.isnull().sum() # no null values

# apply on dataframe
age_mean = data.CLMAGE.mean() # 28.414422241529106, will round off upto 0 decimals.
mean = round(age_mean, 0) # 28.41, 

data['CLMAGE'].fillna(mean, inplace=True)

data.isna().sum() # no missing in CLMAGE column


# using function for binary element columns 

from sklearn.impute import SimpleImputer 
import numpy as np

mode_imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

# using another variable to save the data.
data1 = mode_imputer.fit(data)
data1 = mode_imputer.transform(data)
# or  
data3 = mode_imputer.fit_transform(data)

# check missing values
data3.isna().sum() # will not work since data1 and data3 variables are not dataframe

# making dataframe of either of data1 or data3
data3 = pd.DataFrame(data3)
data3.isna().sum()  # no null values are there, but column name are not there, lets change it.

data3.columns # checking column names

# changing column name 
data3.columns = data.columns
data3.columns # successfully names changed, congratulations rohit. 

###### End of the script ####
