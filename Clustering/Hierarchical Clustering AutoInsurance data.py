# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:20:36 2022

@author: rohit
"""
# Hierarchical Clustering assignment for auto Insurance Data.

# First we will import os library for path

import os
os.chdir(r'D:\360digitmg\lectures\7 CLustring Hierarchy')

# import pandas (data manupulation) to load dataframe.
 
import pandas as pd

# loading excel file not csv.
data = pd.read_csv('Assignment/Autoinsurance.csv ') 
data.columns

data.info()  # 
stats = data.describe()

# check missing data
data.isna().sum() # No missing

# check duplicates
data.duplicated().sum()  # No duplicates

# remove customer column since it not of much useful 
data1 = data.drop('Customer', axis=1)

# Rearrangeing columns to classifyl Numeric and Non-Numerics columns
data1 = data1.iloc[:, [0,2,3,4,5,6,7,9,10,17,18,19,21,22,16,1,8,11,12,13,14,15,20]]
data1.info() # all object column are taken first then numeric columns
# remove data since it is not required, it is objective data hence it is nominal data like customer id 
data1.drop('Effective To Date', axis=1, inplace=True)

# Make data in standard format
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler() # create instance with defalul parameters

df_scale = scaler.fit_transform(data1) # this data has string value variables

# remve string values from this data 
df_scale = scaler.fit_transform(data1.iloc[:,[14,15,16,17,18,19,20,21]]) 
df_scale.describe() # will not work, since it array and not DataFrame

df_scale = pd.DataFrame(df_scale)
stats_scale = df_scale.describe() # mean ~ 0 and std = 1 

# need to add categorical data to scaled data
df_cat = data1.iloc[:, :14]

#  concate data along with columns
df_finale = pd.concat([df_scale, df_cat], axis=1)

# get dummy columns using label encoder
from sklearn.preprocessing import LabelEncoder
lab_encode = LabelEncoder()

df_finale = lab_encode.fit_transform(df_finale)

# caution: to use Label Encoding, we have split data into input and output.
# since data has no output specified. 
# will use another method to get dummy variable.
# in 'OneHotEncoding', dummy column will have names like 0, 1,2,3,... I have reamne them manually
# hence use dummy variables, in this dummy variables will get names automatically.

df_finale = pd.get_dummies(df_finale, drop_first=True)


# Now will try ploting differet dendrogram for optimum clustering size 
from scipy.cluster.hierarchy import linkage, dendrogram

# Linkag = 'single', distance formula = 'Euclidean'
single_lin = linkage(df_finale, method='single', metric='euclidean')

# plot graph
from matplotlib import pylab as plt

plt.figure(figsize=(50,30))
plt.title('Hierarchical Clustering Dendrogram with single type linkage')
plt.xlabel('Index')
plt.ylabel('Distance')

# Dendrogram 
dendrogram(single_lin, leaf_rotation = 0, leaf_font_size=10) # font size is 3 and it will be vertical.
plt.show()
