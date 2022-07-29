# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 21:23:49 2022

@author: rohit
"""

# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load the dataset 
data = pd.read_csv("D:/360digitmg/lectures/20 Project FedEx datasets/FedEx/fedex.csv")

data.columns 
data.info()
data.shape  #  (3604175, 15)
stats = data.describe()
stats_cat = data.describe(include=['O'])
# minimum unique values in categorical data is 20 and max is 299. may need to remove this columns.

 
# get categorical variables 
cat_col = [col for col in data.columns if data[col].dtype == 'O']
num_col = [col for col in data.columns if col not in cat_col]

############## E   D   A #############


# missing and duplicates
data.isnull().sum() # lots of null values 
# Actual_Shipment_Time     81602
# Planned_TimeofTravel       547
# Shipment_Delay           81602
# Delivery_Status          81602

data.duplicated().sum() # yes 4 duplicates

#l see unique value

for col in data.columns:
    print(col, '=', data[col].nunique())
    print(col,'=', data[col].unique())
    print(" ")
# Remove year 


# check outliers 
sns.boxplot(data.Year) #whis = 1.5 (default) # no outliers, constant year value.
# plt.boxplot(data.Year) # whis = 1.5 (default)
# sns.boxplot(data.Year, whis=3)
plt.boxplot(data.Month)          
plt.boxplot(data.DayofMonth)          
plt.boxplot(data.DayOfWeek)          
plt.boxplot(data.Actual_Shipment_Time)          
plt.boxplot(data.Planned_Shipment_Time)        
plt.boxplot(data.Planned_Delivery_Time)         
plt.boxplot(data.Planned_TimeofTravel)    
plt.boxplot(data.Shipment_Delay) 
plt.boxplot(data.Delivery_Status)      

# No outliers found in all.

# univariate analysis
sns.histplot(data.Month) # data collected in each month is almost equal.
sns.histplot(data.DayofMonth) # data collected in each day of month is almost equal.
sns.histplot(data.DayOfWeek)# data collected in each day of week is almost equal.
sns.histplot(data.Actual_Shipment_Time) # actual shipment start from 5 AM to midnight. peak is from 5AM to 8PM.  
sns.histplot(data.Planned_Shipment_Time) # most starts at 6 o'clock in the morning.
sns.histplot(data.Planned_Delivery_Time) # most delivery time is from 7AM to 11PM.
sns.histplot(data.Planned_TimeofTravel) # min time is 30 minutes and max is more than 400minutes
sns.histplot(data.Shipment_Delay) # most of the ships reach destination on time or near to planned time on either side.
sns.histplot(data.Delivery_Status) # assuming 0 = delivery on time, most of shipment reach destination on time.

#################### Data preprocessing

# 1. Remove year variable, null, duplicates
data1 = data.drop('Year', axis=1)
data1.drop_duplicates(keep='first', inplace=True)
data1 = data1.dropna(axis=0, )
data.shape # (3604171, 15)
data1.shape #(3522163, 14) we have much data
# confirm
data1.isnull().sum() # nos 
data1.duplicated().sum() # no

# see categories in dependent column i.e. Delivery_status columns
data1.Delivery_Status.value_counts() # there are two categories. 

################ Unsupervised Learning #################################
# since output has two groups we can use Non-Hierarchical clustering here.
data1.info() 
# Removing categorical data columns and Year column
num_col[0]
num_col[11]
data3 = data1[num_col[1:11]]
data3.columns

# Scaling 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_ss = pd.DataFrame(scaler.fit_transform(data3))
stats_ss = data_ss.describe()

# apply PCA on the data
from sklearn.decomposition import PCA
pca = PCA(n_components=10) # compenents should be equal to number of original columns.
pca_value = pca.fit_transform(data_ss)

pca.components_
pca.components_[0] # First row.

# Amount of variance that each variance explain is
var = pca.explained_variance_ratio_
var

# take cumulative sum of variance
var1 = np.cumsum(np.round(var, decimals=4)*100) #taking % of variance (in cumulative)
var1  #  decimal=4 means e.g. 24.13 ( total digit are 4)

# first  7 PC's giving > 96% inforamtion.
# variance plot for PCA component obtained.
# graph is opposite to scree plot or elbo curve (used in K means clustering)
plt.plot(var1, color='red')
plt.xlabel("Number of PCA's")
plt.ylabel("Information in percentage")
plt.title('Variance Plot')
plt.show()
# from plot, 6PCA giving much information, but i will take 7 PCAs for more accuracy.

# PCA score 
pca_value # convert var1 from array to dataframe

pca_data = pd.DataFrame(pca_value) # No column names

# Give names to columns
pca_data.columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']

# plot 
pca_data.plot(x='PC1', y='PC2', kind='scatter', figsize=(15,8)) # No any co-relation. 

# pca_data is the DataFrame obtained from PCA. Now we will use this data with first seven columns 
# for clustering  K-means clustering. 

pc7 = pd.DataFrame(pca_data.iloc[:,:7]) # considering only first seven columns as mentioned in question.

pc7.var(axis=0)==0 # No zero variance.

##########################################################################
######################## K-Means clustering ############################## 
##########################################################################
# lets assume than cluster_size= 2 (since Kmeans clustered in 3 clusters)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(pc7)

kmeans.labels_

cluste_pc_kmeans = pd.Series(kmeans.labels_)

# add this to wine data
data1['Kmeans_cluster'] = cluste_pc_kmeans

# see accuracy consider Delivery as actual and kmeans_cluster as prediction 
accuracy_kmeans = np.mean(data1.Delivery_Status == data1['Kmeans_cluster']) 
print(accuracy_kmeans)
data1.info()
# since we have drop null and duplicates, the following loop will not work
data1.reset_index(inplace=True)
data1.columns # remove old index col
data1.drop('index', axis = 1, inplace=True)


###############################################################
################# supervised learning #########################
###############################################################
# will take KNN model 

# since the data is already scaled but the data has no ouput col.
# add the output col to scaled data
data_ss['Delivery_Status'] = data1['Delivery_Status']
# Declare label and features
X = data_ss.iloc[:, data_ss.columns != 'Delivery_Status']
y =  data_ss['Delivery_Status']

# split data into training and testing set

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# train the model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 21)
# randomly taking k value = 21.
knn.fit(X_train, Y_train)

pred_test = knn.predict(X_test)

# accuracy 
accuracy_test = np.mean(Y_test == pred_test)
print('The accuracy of testing set is {}.'.format(round(accuracy_test*100, 2)))

# predition with training data
pred_train = knn.predict(X_train)


# accuracy on training data
accuracy_train = np.mean(Y_train == pred_train)
print('The accuracy of traing set is {}.'.format(round(accuracy_train*100, 2)))

# both the accuracy are:
print('The accuracy of traing set is {}.'.format(round(accuracy_train*100, 2)))
print('The accuracy of testing set is {}.'.format(round(accuracy_test*100, 2)))

# Model is perfect, both accuracies are almost same.


###############################################################################
######################### word cloud for categorical data #####################
###############################################################################

# take columns from data1: Carrier_Name, Source, Destination.

words_list = [word for word in data1['Carrier_Name']]
words_list2 = [word for word in data1['Source']]
words_list3 = [word for word in data1['Destination']]

all_words = [words_list+words_list2+ words_list3]
all_words = [word for word in all_words] 
print(all_words[0])
     
# joining all the reviews into into single paragraph
ip_rev_string1 = " ".join(words_list)
ip_rev_string2 = " ".join(words_list2)
ip_rev_string3 = " ".join(words_list3)

ip_rev_string = ip_rev_string1+ip_rev_string2+ip_rev_string3


import nltk
# words the contained in categorical variables
ip_reviews_words = ip_rev_string.split(" ") 

# Tfidf
from sklearn.feature_extraction.text import TfidfVectorizer 
# vectorizer = TfidfVectorizer( ip_reviews_words, use_idf=True, ngram_range=(1,3)) # Not working
vectorizer = TfidfVectorizer( use_idf=True, ngram_range=(1,3))
X = vectorizer.fit_transform(ip_reviews_words)


# joining all words into a paragraph
in_rev_string = " ".join(ip_reviews_words)  
 
# wordcloud can be performed on the string input
# carpus level word cloud
from wordcloud import WordCloud

wordcloud_ip = WordCloud(background_color='White',
                         width=1800,
                         height=1400).generate(ip_rev_string)   
     
plt.imshow(wordcloud_ip) # cloud with unigrams.


############ End of the project ##############





