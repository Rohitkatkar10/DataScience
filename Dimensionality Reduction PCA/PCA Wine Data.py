# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:42:04 2022

@author: rohit
"""

## Principal Component Analysis (PCA)
# Here we wil compare clustering result with PCA and without PCA.
# First will create clustering using Hierarchical and Non-Hierarchical (K-Means) Clustering.

# import required libraries 
import os # to set woriking directory
import pandas as pd   # for data manipulation
import numpy as np # for numeric operation
import matplotlib.pyplot as plt  # for visualization
import seaborn as sns  # for advanced visulization

os.chdir('D:/360digitmg/lectures/9 Dimension Reduction PCA')

wine = pd.read_csv('Assignment/Datasets_PCA/wine.csv')

wine.head()
wine.shape # 178x14. 14 variables are there

wine.columns

# check info
wine.info()   # All Numeric data.

# business decision
wine_stats = wine.describe()

# check null
wine.isna().sum() # No null

# check duplicates
wine.duplicated().sum() # No duplicates
 # check variance of data
wine.var(axis=0) == 0 # No zero variance

# check 'Type' variable to unique value
wine['Type'].unique()
wine['Type'].var()  # will remove this column 

# making copy of data in case if we need original data
wine2 = wine.copy() # default, deep=True, any changes to either  of the data will not reflect other data.
wine.drop('Type', axis=1, inplace=True)

# apply winsorization directly. since cannot check box plot outliers for 14 variables.
from feature_engine.outliers import Winsorizer

winsor = Winsorizer( capping_method = 'iqr', tail = 'both',fold = 1.5,
                    variables = ['Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols',
                           'Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color', 'Hue',
                           'Dilution', 'Proline'])
wine = winsor.fit_transform(wine)

# standardized data in necessary in clustering and PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
wine_ss = pd.DataFrame(scaler.fit_transform(wine))  # column names are numbers.
wine_ss.columns = wine.columns # for Hierarchical
wine_ss_Km = wine_ss.copy()    # for KMeans
wine_ss_pca = wine_ss.copy()   # for PCA


### Hierarchical Clustering
# Dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage
# use Average linkage
lin = linkage(wine_ss, method='average', metric='euclidean')

plt.figure(figsize=(50,30))
plt.title('Hierarchical Clustering Dendrogram with average type linkage')
dendrogram(lin, leaf_rotation=0, leaf_font_size=10)
plt.show()

# cluster size = 5 would be better.
from sklearn.cluster import AgglomerativeClustering
# instance 
instance = AgglomerativeClustering(n_clusters=5, affinity='euclidean',linkage='average').fit(wine_ss)

instance.labels_
cluster = pd.Series(instance.labels_)
cluster.value_counts()

# adding cluster to original data

wine['cluster_hierarchy'] = cluster

#######  K Means clustering ###

# scree plot
TWSS = [] # Total within sum of squares
k = list(range(2,9))

from sklearn.cluster import KMeans
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(wine_ss)
    TWSS.append(kmeans.inertia_)

print(TWSS)

# plot scree plot
plt.plot(k, TWSS, 'ro-')
plt.xlabel('Number of clusters')
plt.ylabel('TWSS')
plt.title('Scree plot')
plt.show()

# cluster size = 3 ( high converegence upto 3)
kmeans = KMeans(n_clusters=3)
kmeans.fit(wine_ss)

kmeans.labels_

cluster_k = pd.Series(kmeans.labels_)

# add this to wine data frame
wine['cluster_kmeans'] = cluster_k


#########################  P C A  #########################

# since the data is already pre-processed and standardized, proceed directly to PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=13) # compenents should be equal to number of original columns.
pca_value = pca.fit_transform(wine_ss)

pca.components_
pca.components_[0] # First row.

# Amount of variance that each variance explain is
var = pca.explained_variance_ratio_
var

# take cumulative sum of variance
var1 = np.cumsum(np.round(var, decimals=4)*100) #taking % of variance (in cumulative)
var1  #  decimal=4 means e.g. 24.13 ( total digit are 4)

# first 9 PC's giving 94% inforamtion.
# But as mentioned in quenstion, we will take 3 PC's.

# variance plot for PCA component obtained.
# graph is opposite to scree plot or elbo curve (used in K means clustering)
plt.plot(var1, color='red')
plt.xlabel("Number of PCA's")
plt.ylabel("Information in percentage")
plt.title('Variance Plot')
plt.show()

# PCA score 
pca_value # convert var1 from array to dataframe

pca_data = pd.DataFrame(pca_value) # No column names

# Give names to columns
pca_data.columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13']

# plot 
pca_data.plot(x='PC1', y='PC2', kind='scatter', figsize=(15,8)) # No any co-relation. 

# pca_data is the DataFrame obtained from PCA. Now we will use this data with first three columns 
# for clustering (both). then will compare these with earlier results.

pc3 = pd.DataFrame(pca_data.iloc[:,:3]) # considering only first three columns as mentioned in question.

pc3.var(axis=0)==0 # No zero variance.

# Hierarchical clustering on PCA data 

lin_pc = linkage(pc3, method='average', metric='euclidean')

plt.figure(figsize=(50,30))
plt.title('dendrogram')

dendrogram(lin_pc, leaf_rotation = 0, leaf_font_size = 5)
plt.show()

# cluster_size = 5 ( from dendrogram)

instance_pc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='average').fit(pc3)

instance_pc.labels_

cluster_pc = pd.Series(instance_pc.labels_)

# add this to wine data.
wine['cluster_pc_hierarchy']= cluster_pc


# KMeans clustering for PCA data
Twss = []
k = list(range(2,9))

for i in k:
    kmeans_pc = KMeans(n_clusters=i)
    kmeans.fit(pc3)
    Twss.append(kmeans.inertia_)
    
# scree plot 
plt.plot(k, Twss, 'ro-')
plt.xlabel('Number of clusters')
plt.ylabel('TWSS')
plt.title('Scree plot')
plt.show()

# Here we are not getting any converegence TWSS is constant for all K values.

# lets assume than cluster_size= 3 (since Kmeans clustered in 3 clusters)
kmeans = KMeans(n_clusters=3)
kmeans.fit(pc3)

kmeans.labels_

cluste_pc_kmeans = pd.Series(kmeans.labels_)

# add this to wine data
wine['cluste_pc_kmeans'] = cluste_pc_kmeans

# we added all the four results of hierarchical and kmeans with and without  pca to wine dataframe.

#################################################################################
#################################################################################
# Taking PCA data upti 90 % accuracy. will try KMeans clustering.

pc8 = pd.DataFrame(pca_data.iloc[:,:8]) # taking 8 columns so we can get 90% accuracy.

Twss8 = []
k8 = list(range(2,9))

for i in k8:
    kmeans_pc = KMeans(n_clusters=i)
    kmeans.fit(pc8)
    Twss8.append(kmeans.inertia_)


plt.plot(k8, Twss8, 'ro-')
plt.xlabel('Number of clusters')
plt.ylabel('TWSS')
plt.title('Scree plot')
plt.show()

# Either we take 3 PC's or 8 PC's it is giving scree plot as constant. 
########################################################################################
######################################################################################## 

# we will take means of data columns and group by each cluster type.
mean_hierarchy = wine.iloc[:,:13].groupby(wine.cluster_hierarchy).mean()
mean_kmeans = wine.iloc[:,:13].groupby(wine.cluster_kmeans).mean()
mean_pc_hierarchy = wine.iloc[:,:13].groupby(wine.cluster_pc_hierarchy).mean()
mean_pc_kmeans = wine.iloc[:,:13].groupby(wine.cluste_pc_kmeans).mean()


# see analysis in document.

############ End of the script ##########333