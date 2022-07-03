# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:00:53 2022

@author: rohit
"""

# Assignemnt for K-Means Clustering on AutoInsurance Data.

# import required libraries 
import os # to set woriking directory
import pandas as pd   # for data manipulation
import numpy as np # for numeric operation
import matplotlib.pyplot as plt  # for visualization
import seaborn as sns  # for advanced visulization

os.chdir('D:/360digitmg/lectures/8 K means Non hierarchy')

insur_data = pd.read_csv('Assignment/Datasets_Kmeans/AutoInsurance (1).csv')
insur_data.head()
insur_data.columns
insur_data.shape  # 9134x24

# see info
insur_data.info()

# check null
insur_data.isna().sum() # Np any null value

# check duplicates
insur_data.duplicated().sum()  # No duplicates

# check business decisions
stats = insur_data.describe()

# 'customer' and 'effective to date' will not give any valuable insight
# hence remove them 
insur_data.drop(['Customer', 'Effective To Date'], axis=1, inplace=True)

insur_data.State.value_counts() # only five states are there.

# divide data into Numeric and Non-Nemeric category.
num_data = insur_data.iloc[:,[1,7,10,11,12,13,14,19]] # both interger and float type.
num_col = [1,7,10,11,12,13,14,19]

cat_col = [x for x in list(range(0,22)) if x not in num_col]
cat_data = insur_data.iloc[:, cat_col]
cat_data2 = cat_data.copy(deep=True)  # default deep=True

# Deep copy = does not reflect changes in copy df to original df.
# shallow copy (deep=False) = reflects changes to original DataFrame.


# get stats
cat_stats = cat_data.describe() # mode and its variable name, number of unique values.
num_stats = num_data.describe()

# Note: 
# Here categorical/discrete data need to convert into labels using label encoding.
# And for Numeric data, see if there is outliers.
# then varibles have different units, normilze it.

# see outliers
num_data.columns
sns.boxplot(x=num_data['Customer Lifetime Value'])  # box and whiskers will be horizontal
sns.boxplot(y=num_data['Customer Lifetime Value'])  # box and whiskers will be Vertical
# check for all variables
sns.boxplot(y=num_data['Customer Lifetime Value'], whis = 3) # Yes 
sns.boxplot(y=num_data['Income'], whis=3)  # No
sns.boxplot(y=num_data['Monthly Premium Auto'], whis = 3) # yes
sns.boxplot(y=num_data['Months Since Last Claim'], whis = 3) # NO
sns.boxplot(y=num_data['Months Since Policy Inception'], whis = 3)   # No
sns.boxplot(y=num_data['Number of Open Complaints'], whis = 3)

num_data['Number of Open Complaints'].value_counts() 
# here 0 number of cases are more, but these are number of complaints, hence will not consider this as outliers.

sns.boxplot(y=num_data['Number of Policies'], whis = 3)  # no  
sns.boxplot(y=num_data['Total Claim Amount'], whis = 3)  #yes
  

# Outliers Treatment (replace outliers use Winsorize)
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both', fold=3,
                    variables=['Customer Lifetime Value', 'Monthly Premium Auto', 'Months Since Policy Inception', 'Total Claim Amount'])

num_data = winsor.fit_transform(num_data)

sns.boxplot(y=num_data['Total Claim Amount'], whis = 3)  #no
sns.boxplot(y=num_data['Customer Lifetime Value'], whis = 3) # NO
sns.boxplot(y=num_data['Monthly Premium Auto'], whis = 3) # No

# univatiate analysis on categorical data
cat_data.columns
plt.hist(cat_data.State)                # california and oregon  are more.
plt.hist(cat_data['Response'])          # No is more
plt.hist(cat_data['Coverage'])          # basic is more
plt.hist(cat_data['Education'])         # bachlor, college, school are more
plt.hist(cat_data['EmploymentStatus'])  # employed are more
plt.hist(cat_data['Gender'])            # both equal
plt.hist(cat_data['Location Code'])     # suburban is more
plt.hist(cat_data['Marital Status'])    # married are more
plt.hist(cat_data['Policy Type'])       # personal auto 
plt.hist(cat_data['Policy'])            # carporate L3 more 
plt.hist(cat_data['Renew Offer Type'])  # offer 1 is more
plt.hist(cat_data['Sales Channel'])     # agent is more
plt.hist(cat_data['Vehicle Class'])     # four-door car is more
plt.hist(cat_data['Vehicle Size'])      # Midsize is more.

# use label encoding on categorical data
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

# since label encoding does not give name to columns after labeling, have to write following code for every variable.
# cat_data['State'] = encoder.fit_transform(cat_data['State'])

# by using loop
for column in cat_data.columns:
    cat_data[column] = encoder.fit_transform(cat_data[column])
    
# Normalize the data. using StandardScalar
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

ss_data = scaler.fit_transform(num_data) # it is array
ss_data = pd.DataFrame(ss_data) # No names to the columns
ss_data.columns = num_data.columns
ss_stats = ss_data.describe()

# Now we have done with the preprocessing .
# now join cat_data and ss_data as one dataframe
data = pd.concat([cat_data, ss_data], axis=1)

# scree plot 
TWSS = []   # total within sum of squares
k = list(range(2,10))

# import Kmeans 
from sklearn.cluster import KMeans

# to decide optimum cluster size
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(data)
    TWSS.append(kmeans.inertia_)
    
# scree plot
plt.plot(k, TWSS, 'bo-')
plt.xlabel('cluster size')
plt.ylabel('Total within sum of squares')
plt.title('scree plot')
plt.show()

# cluster size = 5 
kmeans = KMeans(n_clusters = 5)
kmeans.fit(data)

kmeans.labels_
cluster = pd.Series(kmeans.labels_)

# add this cluster series to 'num_data' and 'insur_data'.
num_data['cluster'] = cluster
cat_data2['cluster'] = cluster
insur_data['cluster'] = cluster

# take mean
mean = num_data.iloc[:,:8].groupby(num_data.cluster).mean()    # get mean values of numeric data
mode = cat_data2.iloc[:,:8].groupby(cat_data2.cluster).describe()   # most appeared variable in categorical data.
count = num_data.cluster.value_counts()

############ End of the Script   ######