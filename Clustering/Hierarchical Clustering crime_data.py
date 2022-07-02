# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:51:02 2022

@author: rohit
"""

# Hierarchical Clustering assignment for Crime Data.

import os
os.chdir(r'D:\360digitmg\lectures\7 CLustring Hierarchy')

# import pandas (data manupulation) to load dataframe.
 
import pandas as pd

# loading CSV file.
data = pd.read_csv('Assignment/crime_data.csv') 
data.head()

# check info
data.info      # shows all rows
data.info()  # count shows no null values. 
data.shape # dimension 50x5

data.columns # here 'Unnamed: 0' columns has Name of states in USA.

# Rename the first column
data.rename(columns={'Unnamed: 0': 'USA_states'}, inplace=True)

# check stats 
data.describe()

# check null values 
data.isna().sum() # no
# check duplicates
data.duplicated().sum() # No

#### we will plot dendrogram without standardization of data

# Let's create dendrogram, so we can decide cluster size.
from scipy.cluster.hierarchy import linkage, dendrogram

data1 = data.iloc[:, 1:]

dist = linkage(data1, method='single', metric='euclidean')
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



##### use Normalized data

def norm_fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

df_norm = norm_fun(data1)
df_norm.describe()

dist_norm = linkage(df_norm, method='single', metric='euclidean')

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


### No change in graph I think either data (normalized or not) set is ok.
# I will use data without normalizing 

###### lets check various linkage methods 
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

# From all dendrogram, All dendrogram except single Linkage have same and clear plot as compared to single linkage

# Hence I select Avrage linkage  numbe of cluster will be 3
# Now we will apply Agglomerative clustering 
from sklearn.cluster import AgglomerativeClustering 
instance = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage= 'average').fit(data1)
# select linkage and distance formula that we have selected in dendrogram.


instance.labels_ # each data point is classified into 0 or 1 or 2. 

# converting row to column for better visualization. will use pd.series function.
cluster_labels = pd.Series(instance.labels_)

# now will add the cluster label column to data1 DataFrame
data1['cluster']=cluster_labels # award and cluster column look same here.

# add states column and take cluster as first column
data1['states']=data['USA_states']
data1 = data1.iloc[:,[4,5,0,1,2,3]]


# Aggregate mean of each group
MeanByClusters = data1.iloc[:,2:].groupby(data1.cluster).mean()
data1.cluster.value_counts()


#### there are 16, 20, 14 states in 0,1,2 clusters. 
# overall crime rate in cluster 1 is much less than the rest two. 
# where as 0th cluster has more crime rate. rapes are double and assults more than double compared to cluster1.
#  cluster 2 has highest population mean, still its crime rate is not more than the 0th cluster.
