# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 18:21:26 2022

@author: rohit
"""

# Assignemnt for K-Means Clustering on Telecom Data.

# import required libraries 
import os # to set woriking directory
import pandas as pd   # for data manipulation
import numpy as np # for numeric operation
import matplotlib.pyplot as plt  # for visualization
import seaborn as sns  # for advanced visulization


os.chdir('D:/360digitmg/lectures/8 K means Non hierarchy')
tele = pd.read_excel('Assignment/Datasets_Kmeans/Telco_customer_churn (1).xlsx', sheet_name = 'Telco_Churn')

tele.head()
tele.columns
tele.info()

# check business decisions
stats = tele.describe() # see count variable
# check unique values in count variable
tele.Count.unique()  # only one value i.e. 1 is there in all rows.

# check missing 
tele.isna().sum() # NO missing

# check duplicatedd
tele.duplicated().sum()  # No duplicates

# From observation first three columns are not of much useful.
tele.Quarter.value_counts() # only one values i.e. Q3 in all rows

# remove first three columns
col =['Customer ID','Count','Quarter']
tele.drop(col, axis=1, inplace=True)

# devide data into discrete and continuous data
tele.info()
num_col = [1,2,5,9,21,22,23,24,25,26]
num_data= tele.iloc[:,num_col]

cat_col = [x for x in list(range(0,27)) if x not in num_col]
cat_data = tele.iloc[:,cat_col]

# check dtype 
num_data.info()
cat_data.info() # All ok

# check outliers in numeric data
plt.boxplot(num_data['Number of Referrals']) # yes
plt.boxplot(num_data['Tenure in Months'])
plt.boxplot(num_data['Avg Monthly Long Distance Charges'])
plt.boxplot(num_data['Avg Monthly GB Download']) #yes
plt.boxplot(num_data['Monthly Charge'])
plt.boxplot(num_data['Total Charges'])
plt.boxplot(num_data['Total Refunds']) # yes
plt.boxplot(num_data['Total Extra Data Charges']) # yes
plt.boxplot(num_data['Total Long Distance Charges']) # yes
plt.boxplot(num_data['Total Revenue']) # yes

# winsorize the data.
from feature_engine.outliers import Winsorizer

winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, 
                    variables =['Number of Referrals','Avg Monthly GB Download','Total Refunds','Total Extra Data Charges',
                                'Total Long Distance Charges','Total Revenue'])
num_data_treated = winsor.fit_transform(num_data)

# check
plt.boxplot(num_data_treated['Total Refunds']) 
plt.boxplot(num_data_treated['Total Extra Data Charges']) 
plt.boxplot(num_data_treated['Total Long Distance Charges']) 
plt.boxplot(num_data_treated['Total Revenue'])

# Now check variance of all num data
num_data_treated.var(axis=0) == 0 # 'Total Refunds' and 'Total Extra charged data' have zero variance.

# remove zero varince columns
num_data_treated.drop(['Total Refunds','Total Extra Data Charges'], axis=1, inplace=True)

# Normalize the data
def norm_fun(i):
    x = (i - i.min())/(i.max()-i.min())
    return(x)
num_data1 = norm_fun(num_data)

stats1 = num_data1.describe()

# label categorical data
from sklearn.preprocessing import LabelEncoder 
label = LabelEncoder()

# get empty dataFrame to store.
cat_data1 = pd.DataFrame()
for column in cat_data.columns:
    cat_data1[column] = label.fit_transform(cat_data[column])

# join treated datasets
data = pd.concat([num_data1, cat_data1], axis=1)


# scree plot 
TWSS = []
k = list(range(2,9))

from sklearn.cluster import KMeans
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(data)
    TWSS.append(kmeans.inertia_)
    
# plot
plt.plot(k, TWSS, 'go-')
plt.xlabel('cluster size')
plt.ylabel('Total within sum of squares')
plt.title('scree plot')
plt.show()

# cluster size = 4
kmeans=KMeans(n_clusters=4)
kmeans.fit(data)

kmeans.labels_

cluster = pd.Series(kmeans.labels_)

# add cluster to cat_data and num_data
cat_data['cluster'] = cluster
num_data['cluster'] = cluster

mean = num_data.iloc[:,:11].groupby(num_data.cluster).mean()    # get mean values of numeric data
count = num_data.cluster.value_counts()


############ End of the Script   ######