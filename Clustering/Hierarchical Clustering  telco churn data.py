# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 21:07:47 2022

@author: rohit
"""

# Hierarchical Clustering assignment for Telco_customer_churn Data.

import os
os.chdir(r'D:\360digitmg\lectures\7 CLustring Hierarchy')

# import pandas (data manupulation) to load dataframe.
 
import pandas as pd

# loading CSV file.
data = pd.read_excel('Assignment/Telco_customer_churn.xlsx', sheet_name = 'Telco_Churn') 
data.head()

data.shape # dimension = 7043x30 
data.info()

# check stats
stats = data.describe() # here in count column has constant value. varinace will be zero.


# check null
data.isnull().sum() # no null values
data.dtypes # out of 30, 11 columns are Numeric

# check duplicates
data.duplicated().sum() # No duplications

# check variance
data.var(axis=0)==0 # here for variance axis = 0 stands for columns
# count has zero var hence not usefull.

# for unique values
for column in data.columns:
    print('Column: {} -No. of unique: {}- Unique Values: {}'.format(column, data[column].nunique(), data[column].unique()))
    
# From Here: the first three columns have no usefull info : remove them
#   there are 13 columns have only "yes" , "No" data. or categorical/discrete data.
#  and 4 column of discrete data, who has data other than just Yes and no. 
# rest 9 columns are have Numeric data with no constant value in it. 

# Remover 1st three columns
columns = ['Customer ID','Count','Quarter']
data.drop(columns, axis=1, inplace=True)

# univariate and bivariate analysis for categorical data
# import visualization library
import matplotlib.pyplot as plt

plt.hist(data['State']) # california has more count followed by Oregon
plt.hist(data.Response) # No is dominant
plt.hist(data.Coverage) # basic has more count
plt.hist(data.Education) # bachelor, college and High-School or below has more count.
plt.hist(data.EmploymentStatus) # employed is more
plt.hist(data['Location Code']) # suburban
plt.hist(data['Marital Status']) # Married
plt.hist(data['Policy Type'])  # Personal Auto
plt.hist(data['Policy']) # personal L3 & L2 
plt.hist(data['Renew Offer Type']) # Offer 1 &2
plt.hist(data['Sales Channel'])  #AGent and Branch
plt.hist(data['Vehicle Class']) # Four door car
plt.hist(data['Vehicle Size']) # Mid size


# here our most of the  numeric data is having same scale i.e. currency or charges, 
# Hence NO NEED to Standardize or Normalize the data. 

# ther are many variables with discrete data.  use dummy variables 

data0 = pd.get_dummies(data)                  # dimension  7043x52 
data1 = pd.get_dummies(data, drop_first=True) # 7043x35, 

data1.columns

###### Now we have done preprocessing of data 
# lets plot dendogram 

from scipy.cluster.hierarchy import linkage, dendrogram

# we will try to plot dendrogram with different linkage.
# method = single linkage and metric = 'euclidean'
single_lin = linkage(data1, method='single', metric='euclidean')


plt.figure(figsize=(50,30))
plt.ylabel('Distance')
plt.xlabel('Index')
plt.title('Hierarchical Clustering Dendrogram with single linkage')

dendrogram(single_lin, leaf_rotation = 0, leaf_font_size = 10)
plt.show

##### method = complete linkage and metric = 'euclidean'
complete_lin = linkage(data1, method='complete', metric='euclidean')

plt.figure(figsize=(50,30))
plt.ylabel('Distance')
plt.xlabel('Index')
plt.title('Hierarchical Clustering Dendrogram with complete linkage')

dendrogram(complete_lin, leaf_rotation = 0, leaf_font_size = 10)
plt.show

##### method = Average linkage and metric = 'euclidean'
average_lin = linkage(data1, method='average', metric='euclidean')

plt.figure(figsize=(50,30))
plt.ylabel('Distance')
plt.xlabel('Index')
plt.title('Hierarchical Clustering Dendrogram with Average linkage')

dendrogram(average_lin, leaf_rotation = 0, leaf_font_size = 10)
plt.show

###### method = centroid linkage and metric = 'euclidean'
centroid_lin = linkage(data1, method='centroid', metric='euclidean')

plt.figure(figsize=(50,30))
plt.ylabel('Distance')
plt.xlabel('Index')
plt.title('Hierarchical Clustering Dendrogram with centroid linkage')

dendrogram(centroid_lin, leaf_rotation = 0, leaf_font_size = 10)
plt.show


###### method = weighted linkage and metric = 'euclidean'
weighted_lin = linkage(data1, method='weighted', metric='euclidean')

plt.figure(figsize=(50,30))
plt.ylabel('Distance')
plt.xlabel('Index')
plt.title('Hierarchical Clustering Dendrogram with weighted linkage')

dendrogram(weighted_lin, leaf_rotation = 0, leaf_font_size = 10)
plt.show


# Out of all the dendrograms "Average Linkage" gives fantastic Insight of data. 
# Now quenstion is to select the number of clusters from dendrogram is very difficult one.
# since customer is either churning or not. so select cluster size as 4


# Hence will select Agglomerative clustering 
from sklearn.cluster import AgglomerativeClustering
instance = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='average').fit(data1)

instance.labels_ # see labels 

# convert it to columns
labels = pd.Series(instance.labels_)

# add cluster to dataframe
data['cluster'] = labels
data.head()

# mode of variables group by clusters

mean = data.iloc[:,:27].groupby(data.cluster).mean()
count = data.cluster.value_counts()

################ The end of script #########