# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:55:00 2022

@author: rohit
"""

## Principal Component Analysis (PCA)
# Here we wil compare clustering result with PCA and without PCA.
# First will create clustering using Hierarchical and Non-Hierarchical (K-Means) Clustering.

# visit following website to know more about the data.
#  https://archive.ics.uci.edu/ml/datasets/heart+disease


# import required libraries 
import os # to set woriking directory
import pandas as pd   # for data manipulation
import numpy as np # for numeric operation
import matplotlib.pyplot as plt  # for visualization
import seaborn as sns  # for advanced visulization

os.chdir('D:/360digitmg/lectures/9 Dimension Reduction PCA')

heart = pd.read_csv('Assignment/Datasets_PCA/heart disease.csv')

heart.info()

# make copy of data (in case if we need it)
heart2 = heart.copy()

# check Missing 
heart.isnull().sum() # No missing

# check duplicates
heart.duplicated().sum() # one duplucate

# Remove duplicate
heart.drop_duplicates(inplace=True)  # No duplicate now 

# Business decisions
stats = heart.describe() # data is in different scale. 

# apply winsorization directly. since cannot check box plot outliers for 14 variables.
from feature_engine.outliers import Winsorizer

winsor = Winsorizer( capping_method = 'iqr', tail = 'both',fold = 1.5,
                    variables = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                           'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'])
heart = winsor.fit_transform(heart)
# proceed assuming that there is no outliers in the given data.

# since for both clustering and PCA, data need to be in one scale.
# make data normalized. 

def norm_fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

heart_ss = norm_fun(heart)

# stats 
stats_ss = heart_ss.describe()

heart_ss.var(axis=0) == 0 # no zero variance. 

# Remove fbs column. it contains all zero after outliers Treatment.
heart_ss.drop('fbs', axis=1, inplace=True)

# Now data is perfect. 
#######  Hierarchical Clustering #####

from scipy.cluster.hierarchy import linkage, dendrogram
# use linkage = average and distance formula = euclidean.

lin = linkage(heart_ss, method='average', metric='euclidean')

# dendrogram plotting
plt.figure(figsize=(50,30))
plt.title('dendrogram')

dendrogram(lin, leaf_rotation=0, leaf_font_size=5)
plt.show()

# decided cluster size = 6

#  clustering
from sklearn.cluster import AgglomerativeClustering
instance = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='average').fit(heart_ss)

instance.labels_

# create series and add this to original datasets.
cluster = pd.Series(instance.labels_)

heart['cluster_hierarchy'] = cluster
heart['cluster_hierarchy'].value_counts()

# K-Means clustering 

# first we will plot scree plot. then will decide cluster size.
twss = [] 
k = list(range(2,9))

from sklearn.cluster import KMeans
for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(heart_ss)
    twss.append(kmeans.inertia_)
    
# if we observe twss list carefully converegence is high upto index3 or index 2. 
# hence cluster size could be 4 or 5.

# lets scree plot to get cluster size exactly.
plt.plot(k, twss, 'ro-')
plt.xlabel('number of clusters')
plt.ylabel('twss')
plt.show()

# cluster size = 4 
kmeans = KMeans(n_clusters=4)
kmeans.fit(heart_ss)
kmeans.labels_

# add this heart dataset
heart['cluster_kmeans'] = pd.Series(kmeans.labels_)

###### PCA #####

from sklearn.decomposition import PCA
pca = PCA(n_components=13)
pca_values = pca.fit_transform(heart_ss)  # this is required PCA dataset, convert into DataFrame.

pca.components_
pca.components_[0] # first row

# Amount of each variance that each variance explain is 
var = pca.explained_variance_ratio_
var

# get cumulative sum of variance 
var1 = np.cumsum(np.round(var, decimals=4)*100)
var1
# getting accuracy upto 95% in 10 PCA.

# varince plot (opposite to scree plot)
plt.plot(var1, color='green')
plt.xlabel('number of PCA')
plt.ylabel('Information in %')
plt.title('Variance plot')
plt.show()
# above plot can also be written in following format.
plt.plot(var1, 'ro-')
plt.xlabel('number of PCA')
plt.ylabel('Information in %')
plt.title('Variance plot')
plt.show()


# select 10 principal components (giving accuracy upto 95%) 
PCA_data = pd.DataFrame(pca_values)
PCA_data.columns # column names are numbers

# give column names 
PCA_data.columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13']

pc10 = PCA_data.iloc[:,:10] # taking only first 10 principal components.


############ Agglomerative Clustring on PCA data #######################

# plot dendrogrma
lin_pc = linkage(pc10, method='average', metric='euclidean')

plt.figure(figsize=(50,30))
plt.title('Dendrogram')

dendrogram(lin_pc, leaf_rotation=0, leaf_font_size=5).fit(pc10)
plt.show()

# from dendrogram cluster size = 6
#clustering 

instance_pc = AgglomerativeClustering(n_clusters=6, affinity='euclidean',linkage='average').fit(pc10)

instance_pc.labels_

# add to original data
heart['cluster_pc_hierarchy'] = pd.Series(instance_pc.labels_)

#### KMeans clustering on PCA data ######## 

twss_pc = []

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(pc10)
    twss_pc.append(kmeans.inertia_)

# scree plot 
plt.plot(k, twss_pc, 'go-')
plt.xlabel('Number of clusters')
plt.ylabel('TWSS')
plt.title('Scree plot')
plt.show()

# from scree plot, cluster size=4.
kmeans = KMeans(n_clusters=4)
kmeans.fit(pc10)

kmeans.labels_

# add this to original dataset
heart['cluster_pc_kmeans'] = pd.Series(kmeans.labels_)

##### all clustering done ########################

# we will take means of data columns and group by each cluster type.
mean_hierarchy = heart.iloc[:,:14].groupby(heart.cluster_hierarchy).mean()
mean_kmeans = heart.iloc[:,:14].groupby(heart.cluster_kmeans).mean()
mean_pc_hierarchy = heart.iloc[:,:14].groupby(heart.cluster_pc_hierarchy).mean()
mean_pc_kmeans = heart.iloc[:,:14].groupby(heart.cluster_pc_kmeans).mean()


########### End of the script #############



