# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 15:57:54 2022

@author: rohit
"""

# Hierarchical Clustering assignment for Airline Data.

# First we will import os library for path

import os
os.chdir(r'D:\360digitmg\lectures\7 CLustring Hierarchy')

# import pandas (data manupulation) to load dataframe.
 
import pandas as pd

# loading excel file not csv.
data = pd.read_excel('Assignment/EastWestAirlines.xlsx') 
# here sheet number(first sheet is zero) or sheet name ( in quotes) is not mentioned.
# by default it will open first (i.e. zeroth) sheet 
data.head()

# since we need from data sheet,  mention sheet number or name.
data = pd.read_excel('Assignment/EastWestAirlines.xlsx', sheet_name=1) # or [sheet_name='data'] also accepted. 
# data is name of 2nd sheet. and its index is 1. hence both are ok.
data.head()
# check info of dataframe
data.info() # No null values, 
data.shape # dimension= 3999x12

# get stats
stats = data.describe() # data is not in same scale and is very hard to interpret
# need to convert in same scale (standerdize or Normalize it).
# see "ID#" column stats it is like serial number starts from 1 and expected to end at 4000.
# but its maximum value is 4021. hence it is not regular numbering. in between number are missing
# but it is ID columnn of planes. so it it possible that some planes are not considered here.


# BUT First let check following things.
# see missing values ( sees already but still confirming)
data.isna().sum() # no any

# check duplicates
data.duplicated().sum() # no any

# check variance
data.var(axis=0) == 0  # check variance for columns, all are non zero variables.

# from observation all variables have continuous data points but award? was seen only ) 0 and 1 
# Hence to confirm is binary data column or not.
data['Award?'].value_counts() # this has binary data.

# lets remove and id column and normalize the data.
data1 = data.drop('ID#', axis=1)

# normalize 
def norm_fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return x 
# pass the data, since 'award?' has binary values either remove it or keep it
# it will have same values.
df_norm = norm_fun(data1) # all values are from 0 to 1.
# now check stats
stats_norm = df_norm.describe()

# Let's create dendrogram, so we can decide cluster size.
from scipy.cluster.hierarchy import linkage, dendrogram

# linkage is used measure the distance between two data points, it has 4 times
# dendrogram is a plot to decide optimum number of clusters

dist = linkage(df_norm, method='single', metric='euclidean')
# single linkage measures minimum distance beetwen members of clusters.
# metric = euclidean is euclidean method to calculate distance. 

# dendrogram
from matplotlib import pylab as plt

# first decide the size of figure, this is empty graph.
plt.figure(figsize=(50,30))
plt.ylabel('Distance')
plt.xlabel('Index')
plt.title('Hierarchical Clustering Dendrogram')

# Here with above graph dimension, fig size is in inches 50 inch by 30 inch.
# and default dpi=100 ( dot per inch). the resolution of figure in dpi.
# if dpi value increased then it will give zoom view of graph.
# if data points are more better we keep dpi low.

#### here i have traied combination ####
 
# plt.figure(figsize=(50,30), dpi=10)
# plt.ylabel('Distance')
# plt.xlabel('Index')
# plt.title('Hierarchical Clustering Dendrogram')

dendrogram(dist, leaf_rotation = 0, leaf_font_size = 10)
plt.show

# From dendrogram, I can see there are mere major two classes. I think data is clustered on the basis of Award column.
# and the graph is much completed below 2 braches which makes it difficult to identify branches. 
# 
# Now we will apply Agglomerative clustering 
from sklearn.cluster import AgglomerativeClustering 
instance = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage= 'single').fit(df_norm)
# select linkage and distance formula that we have selected in dendrogram.

instance.labels_ # each data point is classified into 0 or 1. 

# converting row to column for better visualization. will use pd.series function.
cluster_labels = pd.Series(instance.labels_)

# now will add the cluster label column to data1 DataFrame
data1['cluster']=cluster_labels # award and cluster column look same here.

#  take cluster column are first column 
data1 = data1.iloc[:, [11, 0, 1,2,3,4,5,6,7,8,9,10]]

data1.head()

# aggregate mean of each group 
Mean = data1.iloc[:,1:].groupby(data1.cluster).mean()
# we can see here data is clustered based on awards.

# Now we will remove 'Award?' column along with 'ID#' column
data.rename(columns={'ID#':'ID', 'Award?':'Award'}, inplace=True)

# remove the columns 
data2 = data.drop(['ID', 'Award'], axis=1)

# since the data ha no null, duplicates. we will normalize it.
df_norm2 = norm_fun(data2)

# we will try to plot dendrogram with different linkage.
# method = single linkage and metric = 'euclidean'
single_lin = linkage(df_norm2, method='single', metric='euclidean')

plt.figure(figsize=(50,30))
plt.ylabel('Distance')
plt.xlabel('Index')
plt.title('Hierarchical Clustering Dendrogram with single linkage')

dendrogram(single_lin, leaf_rotation = 0, leaf_font_size = 10)
plt.show

##### method = complete linkage and metric = 'euclidean'
complete_lin = linkage(df_norm2, method='complete', metric='euclidean')

plt.figure(figsize=(50,30))
plt.ylabel('Distance')
plt.xlabel('Index')
plt.title('Hierarchical Clustering Dendrogram with complete linkage')

dendrogram(complete_lin, leaf_rotation = 0, leaf_font_size = 10)
plt.show

##### method = Average linkage and metric = 'euclidean'
average_lin = linkage(df_norm2, method='average', metric='euclidean')

plt.figure(figsize=(50,30))
plt.ylabel('Distance')
plt.xlabel('Index')
plt.title('Hierarchical Clustering Dendrogram with Average linkage')

dendrogram(average_lin, leaf_rotation = 0, leaf_font_size = 10)
plt.show

###### method = centroid linkage and metric = 'euclidean'
centroid_lin = linkage(df_norm2, method='centroid', metric='euclidean')

plt.figure(figsize=(50,30))
plt.ylabel('Distance')
plt.xlabel('Index')
plt.title('Hierarchical Clustering Dendrogram with centroid linkage')

dendrogram(centroid_lin, leaf_rotation = 0, leaf_font_size = 10)
plt.show


###### method = weighted linkage and metric = 'euclidean'
weighted_lin = linkage(df_norm2, method='weighted', metric='euclidean')

plt.figure(figsize=(50,30))
plt.ylabel('Distance')
plt.xlabel('Index')
plt.title('Hierarchical Clustering Dendrogram with weighted linkage')

dendrogram(weighted_lin, leaf_rotation = 0, leaf_font_size = 10)
plt.show

# From dendrograms I see 'complete Linkage' gives best visible graph.
# hence will stick to complete linkage, there are 6 different colours in dendrogram. hence we will try clusters 
# from 2 to 6 

# select agglomerative Clustering way to cluster
# import numpy as np
# for cluster in np.arange(2,7,1):    # we can use for loop to decide optimum cluster size 
# but to avoid complexity, I will take cluster size as 6.

instance2 = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage= 'complete').fit(df_norm2)



instance2.labels_ # each data point is classified into 0,1,2,3,4,5. 

# converting row to column for better visualization. will use pd.series function.
cluster_labels2 = pd.Series(instance2.labels_)

# now will add the cluster label column to data1 DataFrame
data2['cluster']=cluster_labels2 # award and cluster column look same here.

#  take cluster column are first column 
data2 = data2.iloc[:, [10, 0, 1,2,3,4,5,6,7,8,9]]
data2['Award']=data['Award'] # adding award column to data2

data2.head()

# aggregate mean of each group 
Mean_cluster = data2.iloc[:,1:].groupby(data2.cluster).mean()
Mean_award= data2.iloc[:,1:11].groupby(data2.Award).mean()
# we can see here data is clustered based on awards.

count_cluster = data2.iloc[:,1:].groupby(data2.cluster).count() 
# 0 = 2596 ; 1 = 14; 2 = 997; 3 = 4; 4 = 9; 5 = 379 =. cluster name = itc count

count_Award = data2.iloc[:,1:].groupby(data2.Award).count() 
# 0 = 2518; 1 =	1481. 0= award not given, 1 = award given and their count.

'''
Business obj: maximize award distribution
Constrain: use of frequent flyer reward and use of airline credit card.

People in cluster zero are less likely to get an award.
 Since they not taking advantage of various scheme provided by the airline management. 


However, people in rest of cluster they can get awards more likely, 
even though their average balance to be eligible is more, they take benefit of other schemes to achieve the target , 
so they  can get award.

'''

# ########## End of the SCRIPT  ############3
