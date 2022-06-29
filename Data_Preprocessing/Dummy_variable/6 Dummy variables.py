# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 12:21:19 2022

@author: rohit
"""

# import all required libraries 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# set working directory 
os.chdir((r'D:\360digitmg\lectures\3 Data preprocessing'))

# load the data 
animal = pd.read_csv('Datasets/animal_category.csv')
animal.head()
animal.columns # index column is already in dataframe


animal = pd.read_csv('Datasets/animal_category.csv', index_col = 0) 
animal.columns # index col removed, indexing starts from 1.
# we can also drop index column


# check info
animal.info() # 30x4, no null values, all object type. 
animal.shape # dimension of data.

# get 1st and 2nd moment business decisions
animal.describe() # all data columns ara non numeric. 
# it will show unique values, mode (top) and its count(freq)

# from data i can say that, type is label and rest are input columns.

# check missing values
animal.isnull().sum() # no null values

# since all data is object type, cannnot find outliers.
# check duplicates 
animal.duplicated().sum() # 18 values are  duplicates
# duplicates are more than 50% of data, hence will not remove since our task
# to make dummy variables.

# plot histogram ( will give count)
sns.histplot(animal.Animals) # lion is 12 times, followed by cat=6 
sns.histplot(animal.Gender)  # male and female are equal 
sns.histplot(animal.Homly)  # equle
sns.histplot(animal.Types)  # type is more.

animal.Animals.unique()

# convert object data into dummy variables
# dummy converts all data variables into binary. 
dummy_data = pd.get_dummies(animal) #30 x 14
dummy_data.columns

# drop first 
dummy_data1 = pd.get_dummies(animal, drop_first=True) # 30X10 



#####  End of the Script. 