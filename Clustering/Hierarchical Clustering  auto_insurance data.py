# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:41:53 2022

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
data.info()

stats = data.describe()

# check null 
data.isna().sum() # no null
 # check duplicates
data.duplicated().sum() # no duplicates

# now will drop unnecessary columns 'customer' and 'effective to date'.
data1 = data.drop(['Customer', 'Effective To Date'], axis=1)
data1.columns
data1.info()

# there are 16 categorical columns and rest are numeric. will diveide them 
# se we can find outliers and do label encoding
df_cat = data1.iloc[:,[0,2,3,4,5,6,8,9,15,16,17,18,20,21]]
df_num = data1.iloc[:,[1,7,10,11,12,13,14,19]]

#  make data into one scale using label encoding. 
from sklearn.preprocessing import LabelEncoder
labEncoder = LabelEncoder()

df_cat.columns

df_cat['State'] = labEncoder.fit_transform(df_cat['State'])
df_cat['Response'] = labEncoder.fit_transform(df_cat['Response'])
df_cat['Coverage'] = labEncoder.fit_transform(df_cat['Coverage'])
df_cat['Education'] = labEncoder.fit_transform(df_cat['Education'])
df_cat['EmploymentStatus'] = labEncoder.fit_transform(df_cat['EmploymentStatus'])
df_cat['Gender'] = labEncoder.fit_transform(df_cat['Gender'])
df_cat['Location Code'] = labEncoder.fit_transform(df_cat['Location Code'])
df_cat['Marital Status'] = labEncoder.fit_transform(df_cat['Marital Status'])
df_cat['Policy Type'] = labEncoder.fit_transform(df_cat['Policy Type'])
df_cat['Policy'] = labEncoder.fit_transform(df_cat['Policy'])
df_cat['Renew Offer Type'] = labEncoder.fit_transform(df_cat['Renew Offer Type'])
df_cat['Sales Channel'] = labEncoder.fit_transform(df_cat['Sales Channel'])
df_cat['Vehicle Class'] = labEncoder.fit_transform(df_cat['Vehicle Class'])
df_cat['Vehicle Size'] = labEncoder.fit_transform(df_cat['Vehicle Size'])

# jush check value  count on random variable
df_cat['Vehicle Class'].value_counts() # there are 6 values.

# check outliers for numerical data 
import matplotlib.pyplot as plt
df_num.columns

plt.boxplot(df_num['Customer Lifetime Value']); plt.title('outliers for Customer value'); plt.show() # yes
plt.boxplot(df_num['Income']); plt.show() # No
plt.boxplot(df_num['Monthly Premium Auto']); plt.show() # yes
plt.boxplot(df_num['Months Since Last Claim']); plt.show() # no
plt.boxplot(df_num['Months Since Policy Inception']); plt.show() # no
plt.boxplot(df_num['Number of Open Complaints']); plt.show() # interval kind of data
plt.boxplot(df_num['Number of Policies']); plt.show() # interval kind of data
plt.boxplot(df_num['Total Claim Amount']); plt.show() # yes

# there only three column which have many outliers 

# lets winsorize the data, because we can't loose data.
from feature_engine.outliers import Winsorizer

winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5,
                    variables = ['Customer Lifetime Value','Monthly Premium Auto','Total Claim Amount'])

df_num1 = winsor.fit_transform(df_num)

plt.boxplot(df_num1['Total Claim Amount']); plt.show()
plt.boxplot(df_num1['Customer Lifetime Value'])

# no outliers now.

# check variance.
df_num1.var(axis=0) == 0 

stats1 = df_num1.describe()

# Now see the data, some column are in amount, month, count, etc 
# No same scale, hence convert all of them into one scale. 
# normalize the data
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_num_norm = norm_fun(df_num1)
stats_norm = df_num_norm.describe()

# now we normalized  data and encoded data, concate both as one DataFrame
df_finale = pd.concat([df_cat, df_num_norm], axis=1) # concate column-wise

# Now plot the dendrogram 
from scipy.cluster.hierarchy import linkage, dendrogram

# will try different linkages 
# linkage='single', distance_formula='euclidean'
# dendrogram with single linkage is not working. hence we will go use next linkage.

# linkage='complete', distance_formula='euclidean'
complete_lin = linkage(df_finale, method='complete', metric='euclidean')

plt.figure(figsize=(50,30))
plt.xlabel('Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram with complete linkage')

# dendrogram
dendrogram(complete_lin, leaf_rotation=0, leaf_font_size=3)
plt.show()


# linkage='average', distance_formula='euclidean'
average_lin = linkage(df_finale, method='average', metric='euclidean')

plt.figure(figsize=(50,30))
plt.xlabel('Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram with average linkage')

# dendrogram
dendrogram(average_lin, leaf_rotation=0, leaf_font_size=3)
plt.show()

# linkage='centroid', distance_formula='euclidean'
centroid_lin = linkage(df_finale, method='centroid', metric='euclidean')

plt.figure(figsize=(50,30))
plt.xlabel('Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram with centroid linkage')

# dendrogram
dendrogram(centroid_lin, leaf_rotation=0, leaf_font_size=3)
plt.show()

# out of all graphs complete linkage gives good insight about the data.
# From Dendrogram I dicide cluster size = 5

# use Agglomerative Clustering 
from sklearn.cluster import AgglomerativeClustering
instance = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete').fit(df_finale)

instance.labels_ # it is in array

# convert array to series
cluster_labels = pd.Series(instance.labels_) 

# add this to data frame 
df_finale['cluster'] = cluster_labels
data1['cluaste'] =  cluster_labels

# Take mean 
mean = data1.iloc[:, :22].groupby(data1.cluaste).mean()



## End Of the Script #########