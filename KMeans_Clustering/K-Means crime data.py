# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 17:05:54 2022

@author: rohit
"""

# Assignemnt for K-Means Clustering on crime Data.

# import required libraries 
import os # to set woriking directory
import pandas as pd   # for data manipulation
import numpy as np # for numeric operation
import matplotlib.pyplot as plt  # for visualization
import seaborn as sns  # for advanced visulization

# set working directory
os.chdir('D:/360digitmg/lectures/8 K means Non hierarchy')

crime_data = pd.read_csv('Assignment/Datasets_Kmeans/crime_data (1).csv')
crime_data.head()
crime_data.info() # 50x5

# check business decisions
stats = crime_data.describe()

# check null 
crime_data.isnull().sum() # No null

# check duplicatess
crime_data.duplicated().sum() # no duplicates

# Remove unnamed column
crime_data.drop('Unnamed: 0', axis=1, inplace=True)
crime_data.columns

# check for outliers box plot
# for column in crime_data.columns:
#     sns.boxplot(crime_data[column])
#     plt.show()                   

# loop to plot a graph does not work, it is plotting all plots into one.

sns.boxplot(crime_data.Murder)     # No 
sns.boxplot(crime_data.Assault)  # No
sns.boxplot(crime_data.UrbanPop)  # no
sns.boxplot(crime_data.Rape)    # yes

# replace outliers
# since outliers are less, will use Winsorizer
from feature_engine.outliers import Winsorizer

winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables = ['Rape'])

data = winsor.fit_transform(crime_data)

sns.boxplot(data.Rape) # No outliers

# check variance
data.var(axis=0) == 0 # all non zero
 
# univariate analysis
plt.hist(data.Murder)
plt.hist(data.Assault)
plt.hist(data.UrbanPop)
plt.hist(data.Rape)


# normalize the data (it is recommended in K-Means)
def Norm_fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

data_norm = Norm_fun(data)
data_norm.describe()

# scree plot 
TWSS = list() # total within sum of squares
k = list(range(2,9))

from sklearn.cluster import KMeans
for i in k:
    kmeans = KMeans(n_clusters = i) # will keep other parameters as defaults
    kmeans.fit(data_norm)
    TWSS.append(kmeans.inertia_)

# scree plot
plt.plot(k, TWSS, 'g*-')
plt.xlabel('Number of CLusters')
plt.ylabel('TWSS')
plt.show()

# rate of convergence happens drastically  from cluster 2 to 4.
# convergence rate becomes less after 4.
# hence cluster size = 4

kmeans = KMeans(n_clusters = 4)
kmeans.fit(data_norm)

kmeans.labels_
# makes it column
cluster = pd.Series(kmeans.labels_)

# add cluster to data
crime_data['Cluster'] = cluster
# make cluster column as first
crime_data = crime_data.iloc[:,[4,0,1,2,3]]

mean = crime_data.iloc[:,1:].groupby(crime_data.Cluster).mean()
count = crime_data.Cluster.value_counts()


### End of the script ####