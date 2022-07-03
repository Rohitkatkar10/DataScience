# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:05:03 2022

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

air_data = pd.read_excel('Assignment/Datasets_Kmeans/EastWestAirlines (1).xlsx', sheet_name='data')

air_data.head()
air_data.columns

# change columns names 
air_data.rename(columns={'ID#':'ID', 'Award?':'Award'}, inplace=True)
air_data.columns

# check info
air_data.info() # 3999x12; No null values, all int data types

# get stats
stats = air_data.describe()

# check missing for confirmation
air_data.isnull().sum() # No Missing
# check duplicates
air_data.duplicated().sum() # No Duplication 

# drop ID column, since it is not giveing any useful information.
data = air_data.drop(['ID'], axis=1)
data.columns

# check outliers 
sns.boxplot(data.Balance) #whis = 1.5 (default) # many outliers, making right skew data.
# plt.boxplot(data.Balance) # whis = 1.5 (default)
# sns.boxplot(data.Balance, whis=3)
plt.boxplot(data.Qual_miles)         # yes outliers 
plt.boxplot(data.cc1_miles)          # No
plt.boxplot(data.cc2_miles)          # yes
plt.boxplot(data.cc3_miles)          # yes
plt.boxplot(data.Bonus_miles)        # yes
plt.boxplot(data.Bonus_trans)        # yes
plt.boxplot(data.Flight_miles_12mo)  # yes
plt.boxplot(data.Flight_trans_12)    # yes
plt.boxplot(data.Days_since_enroll)  # no
plt.boxplot(data.Award)              # no


# will treat outliers with winsorization technique
from feature_engine.outliers import Winsorizer
# create instance
winsor = Winsorizer(capping_method = 'iqr', fold=1.5, tail = 'both',
variables = ['Balance','Qual_miles','cc2_miles','cc3_miles','Bonus_miles','Bonus_trans','Flight_miles_12mo','Flight_trans_12'])

# Alternative method to winsorize the outliers

# from scipy.stats.mstats import winsorize 
# data['Balance']=winsorize(data.Balance,limits=[0.07, 0.093]) # we have to write this for every variables.

# winsorize will consider data between limits from 0.07 to 0.093. or 7 percentile and 93 percentile 
# all the upper values and lower values will replaced by 93 percentile and 7 percentile respectively. 

data = winsor.fit_transform(data)

# now confirm 
sns.boxplot(data.Balance)
plt.boxplot(data.Qual_miles)          
plt.boxplot(data.cc1_miles)         
plt.boxplot(data.cc2_miles)     
plt.boxplot(data.cc3_miles)       
plt.boxplot(data.Bonus_miles)     
plt.boxplot(data.Bonus_trans)        
plt.boxplot(data.Flight_miles_12mo)  
plt.boxplot(data.Flight_trans_12)    
plt.boxplot(data.Days_since_enroll) 
plt.boxplot(data.Award)                  # No outliers in all variables


# now will check variance and will remove zero variance column
data.var(axis=0)==0  # axis=0 is for columns here only.

# three column have zero variance , remove them ( zero varince =  NO information)
col = ['Qual_miles','cc2_miles','cc3_miles']
data.drop(col, axis=1, inplace=True)

# check stats
stats1 = data.describe()


# if you see data carefully, data has different units, some columns are in 'miles', some are in 
# ' Number of transactions', etc. hence we have to make our data as a standard format.
# It is always recommended to use Normolised or standardised data in K-Means clustering.

# will use Normalization 
from sklearn.preprocessing import normalize
data_norm = normalize(data) # it is array
data_norm = pd.DataFrame(data_norm)
stats_norm = data_norm.describe()

# another method

def Norm_fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

data_norm_fun = Norm_fun(data)
stats_norm_fun = data_norm_fun.describe()
# function defined data is perfectly normalized but sklearn.preprocessing.normalize makes imperfect.

# Univariate analysis
plt.hist(data.Balance); plt.show()
plt.hist(data.cc1_miles) 
plt.hist(data.Bonus_miles)
plt.hist(data.Bonus_trans)
plt.hist(data.Flight_miles_12mo)
plt.hist(data.Flight_trans_12)
plt.hist(data.Days_since_enroll) # this seems Normalised data, all above are right skewed.
plt.hist(data.Award)  # less people have got award. 

# Use Scree plot or Elbo plot to decide Number of clusters required.
TWSS = []  # Total  Within Sum of Squares
k = list(range(2,9)) # will try K values from 2 to 8

# import K means 
from sklearn.cluster import KMeans

# create loop to for best K value
for i in k:
    kmeans = KMeans(n_clusters=i)  # instance for kmeans
    kmeans.fit(data_norm_fun)  # fitting normalised data in instance
    TWSS.append(kmeans.inertia_)  # save all inertia to list, can be used to plot scree plot
    
# Now plot scree plot
plt.plot(k, TWSS, 'ro-')  # x axis = k , y axis=TWSS, ro- = [r = red , o=circle, - = line] different combinations can be used. 
# scree plot will have line with red color and at cluster it will show circle. 
plt.xlabel('Number of clusters')
plt.ylabel('Total_Within_SS')
plt.show()

# form scree plot the Total within ss is decreasing as number of clusters increase. 
# we will consider cluster number where convergence happens drastically.
# see plot, from cluster 2 to 3, TWSS drastically changes.
# it(TWSS) reduced from 3 to 4. 
# hence we will consider cluster size = 4.

# Hence, K = 4 as optimum number of clusters
K_model = KMeans(n_clusters = 4)
K_model.fit(data_norm_fun) # fitting normalized data, assuming cluster size as .

K_model.labels_  # Getting labels in row.

cluster = pd.Series(K_model.labels_) # converting to column
# add this to dataframe
data['cluster'] = cluster
data.head()
data.shape

# make cluster column as 0 
data = data.iloc[:,[8,0,1,2,3,4,5,6,7]]



# to save file
# add this cluster column to original air_data
air_data['cluster'] = cluster
air_data = air_data.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]

# take mean grouped by clusters
mean = air_data.iloc[:,1:].groupby(air_data.cluster).mean()
count = air_data.iloc[:,0].groupby(air_data.cluster).count()

# saving
air_data.to_csv('clustered_airline.csv', encoding='utf-8')
os.getcwd()


####### End of the Script #####