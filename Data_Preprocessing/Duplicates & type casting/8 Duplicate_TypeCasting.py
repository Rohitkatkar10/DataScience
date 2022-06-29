# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:50:21 2022

@author: rohit
"""

import os
os.chdir('D:/360digitmg/lectures/3 Data preprocessing/Datasets')

import pandas as pd
data = pd.read_csv('OnlineRetail.csv') # 'uts-8' (default) is not working change encoding to 'latin1'.
data = pd.read_csv('OnlineRetail.csv', encoding='latin1')  # worked 
data.columns # gives names of the columns
data.shape  # data dimension = 541909 x 8

# check info
data.info() # customer ID is nominal data change to object. see description and customer ID : null values are there.

# customer id is float here, but is not useful since it is nominal data hence need to change its data type to object.
data.CustomerID.dtypes # float64
data.CustomerID = data.CustomerID.astype('object') 
data.CustomerID.dtypes # o for object

# check business decisions
data.describe()

# check missing values
data.isna().sum() # description= 1454, customer ID = 135080, rest=0
# data.isnull().sum() # optional to find null values

# check duplicates
data.duplicated().sum() # 5268 duplucates are there.

# first drop the duplicates and then will see missing values
data.drop_duplicates(keep='first', inplace=True)

data.duplicated().sum() # no duplicates

# check missing values again
data.isnull().sum() # no change.
# Missing data in customer ID column is more nearly 20% data is not there,
# we cannot simply remove this data, since data is precious, cannot loose it. 
# but it is customer id,what to impute here. since it is not helping in analsis. it is nominal data.

data.CustomerID.mode()
from scipy import stats 
stats.mode(data.CustomerID) # mode of obervation 17841 is 4035 count. 
# since mode is just 4035 over missing values of over 1,00,000. we cannot impute mode here. since it will make 17841 a absolute majority.
 
# remove the  missing data
data1 = data.dropna()

# check again
data1.isna().sum() # no missing values now

# check  outliers with box plot
import seaborn as sns
sns.boxplot(data1['UnitPrice'])  # Many outliers
sns.boxplot(data1['Quantity']) # many outliers

# remove outliers using winsorization
from feature_engine.outliers import Winsorizer 

winsor  = Winsorizer(capping_method='iqr', tail='both', fold=3, variables=['UnitPrice','Quantity'])
perfect_data = winsor.fit_transform(data1)

sns.boxplot(perfect_data['UnitPrice'])  # Many outliers
sns.boxplot(perfect_data['Quantity']) # many outliers

####  End of the script ###