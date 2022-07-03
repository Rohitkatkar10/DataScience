# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 10:46:30 2022

@author: rohit
"""

# Assignemnt for K-Means Clustering on Airline Data.

# import required libraries 
import os # to set woriking directory
import pandas as pd   # for data manipulation
import numpy as np # for numeric operation
import matplotlib.pyplot as plt  # for visualization
import seaborn as sns  # for advanced visulization

# set working directory
os.chdir('D:/360digitmg/lectures/8 K means Non hierarchy')

insur_data = pd.read_csv('Assignment/Datasets_Kmeans/insurance Dataset.csv')
insur_data.head()
insur_data.columns

insur_data.info()

# business decisions
stats = insur_data.describe() # data has different units like, age in years, income in Rupees, claim in count, etc.

# check null
insur_data.isna().sum() # No missing

# check duplicates 
insur_data.duplicated().sum() # no duplicates

# univariate analysis
sns.histplot(insur_data.Age)
sns.histplot(insur_data['Premiums Paid'])
sns.histplot(insur_data['Days to Renew'])
sns.histplot(insur_data['Claims made'])   # there is outliers here.
sns.histplot(insur_data['Income'])

# check outliers 
plt.boxplot(insur_data['Premiums Paid']) # yes
plt.boxplot(insur_data['Age']) # NO
plt.boxplot(insur_data['Days to Renew']) # no
plt.boxplot(insur_data['Claims made']) # yes
plt.boxplot(insur_data['Income']) # No 


# lets winsorize the outliers instead of deleting it
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['Premiums Paid', 'Claims made'] )

data = winsor.fit_transform(insur_data)
plt.boxplot(data['Claims made']) 
plt.boxplot(data['Premiums Paid']) # no outliers in both

# check variance 
data.var(axis=0) == 0   # No zero variance

# will use Normalization 
def Norm_fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

data = Norm_fun(data)

stats_norm = data.describe()

# scree plot
TWSS = []
k = list(range(2,9))

from sklearn.cluster import KMeans
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(data)
    TWSS.append(kmeans.inertia_)

# scree plot
plt.plot(k, TWSS, 'g*-')
plt.xlabel('Number of clusters')
plt.ylabel('Total_within_SS')
plt.title('Scree Plot')
plt.show()
# Rate of converegence is more upto 4 
# hance cluster size = 4

kmeans = KMeans(n_clusters = 4)
kmeans.fit(data)

kmeans.inertia_

kmeans.labels_ # values are in rows

cluster = pd.Series(kmeans.labels_)

# add this cluster series to dataframe

insur_data['Cluster'] = cluster # original

 
# make cluster column as 0th.
insur_data = insur_data.iloc[:, [5,0,1,2,3,4]]

# take thier mean
mean = insur_data.iloc[:, 1:].groupby(insur_data.Cluster).mean()
count = insur_data.Cluster.value_counts()
