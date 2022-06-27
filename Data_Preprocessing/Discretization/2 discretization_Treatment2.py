# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 06:36:03 2022

@author: rohit
"""

'''
Name - Rohit Katkar
Batch ID- 23122021
'''

# Very first import all the Required libraries to environment.

import os # to set working directory.
import pandas as pd   # to Manupulate data.
import numpy as np    # for basic numeric  operation.
import seaborn as sns # for visualization.

# set working directory
os.chdir(r'D:\360digitmg\lectures\3 Data preprocessing')

# import csv file to Environment
data = pd.read_csv('datasets/Iris.csv') # dataset is folder in current working directory.
data.head()
data.columns # There is a unnamed column in dataframe for indexing but indexing starts from 1 instead of zero.



# since there is a column for indexing in original csv file.
# will make column zero is index column. 
# data = pd.read_csv('datasets/Iris.csv', index_col=0) 
# data.head() # here see index starts from one
# data.columns 

# We will remove the Unnamed column from DataFrame. since we want indexin from zero.

data = data.drop(['Unnamed: 0'], axis=1) # or data.drop(['Unnamed: 0'], axis=1, inplace=True) both works.
data.head()  # will show first five samples.
data.columns # No unnamed column.

# saving copy of data.
data1=data.copy() 

# check dimension and data type 
data.shape  #150 x 5
data.info() # Non-null values and data type.

# Find measure of central tendacy 
data.describe()

# From these statistics, i can say that mean, median, range are well within control.
# there is no much difference between mean and median, hence i can say that 
# mean is not influenced by outliers. 
# from this, i can say that there might not be outliers, but need to check.

# check for missing values 
data.isna().sum() # No null values.

# check for duplicate values.
data.duplicated()       # Boolean output. need to take sum of duplicate values.
data.duplicated().sum() # one duplicate value found.

# remove duplicate values.
data.drop_duplicates(keep='first', inplace=True) # Here row 143 got removed. it throw error in loop.
data.shape # 149 x 5

# check unique vlaues
data['Sepal.Width'].value_counts()       # counts same values.
data['Sepal.Width'].value_counts().sum() # sum of count. 

# using unique function
data['Sepal.Width'].unique()    # will give array of unique values.
data['Sepal.Width'].nunique()   # will five count of unique values.


# ".value_counts()" will values and their count in column in console
# ".value_counts().sum()" will give sum of count of unique values.
  
# BUT

# ".unique()' will give only array of unique values, but will not give its count
  
# ".nunique()" will only count of unique values but not unique values.
  
data['Sepal.Width'].unique().sum()   # this will give sum of unique values
data['Sepal.Width'].sum()            # sum of all 149 samples.

# For all columns. This will help us to find out Non-Numeric values in columns.
data['Sepal.Width'].unique() 
data['Sepal.Length'].unique()
data['Petal.Length'].unique()
data['Petal.Width'].unique()
data['Species'].unique() 

#check outliers using box plot
sns.boxplot(data['Sepal.Length'])   # No Outliers
sns.boxenplot(data['Sepal.Width'])  # its a box en plot, similar to box plot.
sns.boxplot(data['Sepal.Width'])    # yes, some outliers.
sns.boxplot(data['Petal.Length'])   # NO
sns.boxplot(data['Petal.Width'])    # no


# outliers in the 'sepal.width' column aare very less and they are near to maximum value, hence will not remove them.

# But to Replace the outliers with max or min values.

#from feature_engine.outliers import Winsorizer

# winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
#                          tail='both', # cap left, right or both tails, 
#                          fold = 1.5,
#                          variables = ['Sepal.Width'])
                          
# Now fit this model to outliers column.

#data=winsor.fit_transform(data) # stored in new variable. this will consider entire data set for fit_transform function.


# sns.boxplot(data['Sepal.Width'])    # Now, No Outliers.


# we will do discritization based on length and width.

# Discretize each column
data['p_len_size'] = pd.cut(x=data['Petal.Length'], bins=[min(data['Petal.Length'])-1, data['Petal.Length'].mean(), max(data['Petal.Length'])],labels=('short petal','lengthy petal'))
data['p_len_size'].value_counts() # there are 92 lengthy and 56 short total 148,one is missing
data['p_len_size'].unique() # there is Nan value.

min(data['Petal.Length'])  # min value is 1.

# The minimum value is one, hence it is  not counting min value while excuting command. Hence will put (min -1 ) in that command.

### Trying Another method.  ##### 

# insert new column in DataFrame at 6th position at last.
data.insert(6, 'Petal_size'," " ) # column counting starts from zero.

# data.drop(['Petal_size'], axis=1, inplace=True) # To remove any column this is the command

# for i in range(1, len(data['Petal.Length'])+1): # the index starts from 1, hence start =1 stop= n-1 (default), hence we will add 1. therefore stop= (150-1)+1.
#    if data['Petal.Length'][i] > data['Petal.Length'].mean() and data['Petal.Width'][i] > data['Petal.Width'].mean(): 
#        data['Petal_size'][i]='Large Petal Species'
#    else:
#        data['Petal_size'][i]='Small Petal Species' 

# here it throws error as 142. since row 142 is not present, it is removed in duplication. 
# In this duplicate removed but its place is not filled by successive row, Hence row 142 is not present there.

# reset the index to dataframe
data.reset_index(inplace=True) # indexing is reasigned from 0 to 148
data.columns # old index column is there with no 142 row. hence remove it.
data.drop(['index'], axis=1, inplace=True) 

for i in range(0, len(data['Petal.Length'])): # the index starts from 1, hence start =1 stop= n-1 (default), hence we will add 1. therefore stop= (150-1)+1.
    if data['Petal.Length'][i] > data['Petal.Length'].mean() and data['Petal.Width'][i] > data['Petal.Width'].mean(): 
        data['Petal_size'][i]='Large Petal Species'
    else:
        data['Petal_size'][i]='Small Petal Species' 


########## Here copy of data (i.e. data1) but without removing duplicate value. 


# insert new column in DataFrame at 5th position at last.
data1.insert(5, 'Petal_size'," " ) # column counting starts from zero.

# data1.drop(['Petal_size'], axis=1, inplace=True) # To remove any column this is the command

for i in range(0, len(data1['Petal.Length'])): # check index start, if starts from 1 then start =1. here starts from zero.
    if data1['Petal.Length'][i] > data1['Petal.Length'].mean() and data1['Petal.Width'][i] > data1['Petal.Width'].mean(): 
        data1['Petal_size'][i]='Large Petal Species'
    else:
        data1['Petal_size'][i]='Small Petal Species' 


# insert new column in DataFrame at 6th position at last.
data1.insert(6, 'Sepal_size'," " ) # column counting starts from zero.

# data1.drop(['Sepal_size'], axis=1, inplace=True)

for i in range(0, len(data1['Sepal.Length'])): # the index starts from 0.
    if data1['Sepal.Length'][i] > data1['Sepal.Length'].mean() and data1['Sepal.Width'][i] > data1['Sepal.Width'].mean(): 
        data1['Sepal_size'][i]='Large Sepal Species'
    else:
        data1['Sepal_size'][i]='Small Sepal Species' 


   
# Creating new column to decide species size.

data1.insert(7, 'species_size', " ")
    
for i in range(0, len(data1['Sepal.Length'])):
    if data1['Petal_size'][i] == 'Large Petal Species' and data1['Sepal_size'][i] == 'Large Sepal Species':
        data1['species_size'][i]='Large'
    elif data1['Petal_size'][i] == 'Large Petal Species' and data1['Sepal_size'][i] == 'Small Sepal Species' or data1['Petal_size'][i] == 'Small Petal Species' and data1['Sepal_size'][i] == 'Large Sepal Species':
        data1['species_size'][i]='Medium'   
#    elif data1['Petal_size'][i] == 'Small Petal Species' and data1['Sepal_size'][i] == 'Large Sepal Species':
#        data1['species_size'][i]='Medium' 
    else:
        data1['species_size'][i]='Small' 
        
        
data1['species_size'].value_counts() # hence we have 64 Medium, 61 small, 25 large. 


# saving the file.
data1.to_csv('Iris_size.csv', encoding='utf-8')
# find where is it saved
os.getcwd() #  get current working directory.
 
######### End of the Script. ###########
