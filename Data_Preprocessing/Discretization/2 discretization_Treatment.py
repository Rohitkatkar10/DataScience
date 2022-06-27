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
import pandas as pd # to Manupulate data.
import numpy as np # for basic numeric  operation.
import seaborn as sns # for visualization.

# set working directory
os.chdir(r'D:\360digitmg\lectures\3 Data preprocessing')

# import csv file to Environment
data1 = pd.read_csv('datasets/Iris.csv') # dataset is folder in current working directory.
data1.head()
data1.columns

# there is a unnamed column for indexing will remove that using drop function or we can use it as index column.
# first will remove it.
check = data1.drop(['Unnamed: 0'], axis=1)
check.head() # index starts from zero. 
check.columns

# since there is a column for indexing in original csv file.
# will make column zero is index column. 
data = pd.read_csv('datasets/Iris.csv', index_col=0) 
data.head() # here see index starts from one
data.columns 



# check dimension and data type 
data.shape  #150 x 5
data.info()

# Find measure of central tendacy 
data.describe()

'''
From these statistics, i can say that mean, median, range are well within control.
there is no much difference between mean and median, hence i can say that 
mean is not influenced by outliers. 
from this, i can say that there might not be outliers, but need to check.

'''
# check for missing values 
data.isna().sum() # No null values.

# check for duplicate values.
data.duplicated() # Boolean output. need to take sum of duplicate values.
data.duplicated().sum() # one duplicate value found.

# remove duplicate values.
data.drop_duplicates(keep='first', inplace=True) # Here row 143 got removed. it throw error in loop.
data.shape # 149 x 5

# check unique vlaues
data.value_counts() # output too big to see, will save to variable
unique_data=data.value_counts() # this is useless. 

'''
Above method will value counts for whole data, we cannot interpret it.
hence will count values column wise.

to count value ".unique()" and ".nunique()" also used.
'''
data['Sepal.Length'].value_counts()  # all are float values
data['Sepal.Width'].value_counts() 
data['Sepal.Width'].value_counts().sum() 

# using unique function
data['Sepal.Width'].unique()       # will give array of unique values.
data['Sepal.Width'].nunique()   # will only count or sum of unique values.

'''
".value_counts()" will values and their count in column in console
".value_counts().sum()" will give sum of count of unique values.
  
BUT
".unique()' will give only array of unique values, but will not give its count
  
".nunique()" will only count of unique values but not unique values.
  
'''

 data['Sepal.Width'].unique().sum() # this will give addition of unique values
 data['Sepal.Width'].sum()            # sum of all 149 samples.
 
data.columns
data['Sepal.Width'].unique() 
data['Sepal.Length'].unique()
data['Petal.Length'].unique()
data['Petal.Width'].unique()
data['Species'].unique() 

'''
From abov outputs I can see no abnornal value in it.
'''
 
#check outliers 
sns.boxplot(data['Sepal.Length'])  # No Outliers
sns.boxenplot(data['Sepal.Width']) 
sns.boxplot(data['Sepal.Width'])    # yes, some outliers.
sns.boxplot(data['Petal.Length'])   # NO
sns.boxplot(data['Petal.Width'])  # no


#remove the outliers.

#from feature_engine.outliers import Winsorizer

# winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
#                          tail='both', # cap left, right or both tails, 
#                          fold = 1.5,
#                          variables = ['Sepal.Width'])
                          
# now fit this model to outliers column.

#data=winsor.fit_transform(data) # stored in new variable.

# sns.boxplot(data['Sepal.Width'])    # yes, some outliers


'''
outliers in the 'sepal.width' column aare very less and they are near to maximum value
hence will not remove them.
'''

'''
From the data sepal and  petal and their length and width are responsible for 
species category. 
hence we will do discetization based on species.
and we have many unique values in length and width columns but only 3 values in 
species column.
'''

# Discretize each column
data['p_len_size'] = pd.cut(x=data['Petal.Length'], bins=[min(data['Petal.Length'])-1, data['Petal.Length'].mean(), max(data['Petal.Length'])],labels=('short petal','lengthy petal'))
data['p_len_size'].value_counts() # there are 92 lengthy and 56 short total 148,one is missing
data['p_len_size'].unique() # there is Nan value.

min(data['Petal.Length'])  # min value is 1.

'''
the minimum value is one, hence it is  not counting min value while excuting command.
hence will put (min -1 ) in that command.

''' 

# insert new column in DataFrame at 6th position at last.
data.insert(6, 'Petal_size'," " ) # column counting starts from zero.

# data.drop(['Petal_size'], axis=1, inplace=True) # To remove any column this is the command

for i in range(1, len(data['Petal.Length'])+1): # the index starts from 1, hence start =1 stop= n-1 (default), hence we will add 1. therefore stop= (150-1)+1.
    if data['Petal.Length'][i] > data['Petal.Length'].mean() and data['Petal.Width'][i] > data['Petal.Width'].mean(): 
        data['Petal_size'][i]='Large Petal Species'
    else:
        data['Petal_size'][i]='Small Petal Species' 
        
# here it throws error as 143. since row 143  is not present, it is removed in duplication. 
#In this duplicate removed but its place is not filled by successive row, Hence row 143 is not present there.



# we will use check DataFrame to discritize the data. 

# check for duplicate values.
check.duplicated() # Boolean output. need to take sum of duplicate values.
check.duplicated().sum() # one duplicate value found.

# In this duplicate removed but its place is filled by successive row. 

check.insert(5, 'Petal_size'," " ) # column counting starts from zero.

#check.drop(['Petal_size'], axis=1, inplace=True)

for i in range(0, len(check['Petal.Length'])): # the index starts from 0.
    if check['Petal.Length'][i] > check['Petal.Length'].mean() and check['Petal.Width'][i] > check['Petal.Width'].mean(): 
        check['Petal_size'][i]='Large Petal Species'
    else:
        check['Petal_size'][i]='Small Petal Species' 
# No error thrown here. 


# insert new column in DataFrame at 6th position at last.
check.insert(6, 'Sepal_size'," " ) # column counting starts from zero.

# check.drop(['Sepal_size'], axis=1, inplace=True)

for i in range(0, len(check['Sepal.Length'])): # the index starts from 0.
    if check['Sepal.Length'][i] > check['Sepal.Length'].mean() and check['Sepal.Width'][i] > check['Sepal.Width'].mean(): 
        check['Sepal_size'][i]='Large Sepal Species'
    else:
        check['Sepal_size'][i]='Small Sepal Species' 
    
# Creating new column to get large species.

check.insert(7, 'species_size', " ")
    
for i in range(0, len(check['Sepal.Length'])):
    if check['Petal_size'][i] == 'Large Petal Species' and check['Sepal_size'][i] == 'Large Sepal Species':
        check['species_size'][i]='Large Species'
    else:
        check['species_size'][i]='Small Species' 
        
        
check['species_size'].value_counts() # hence we have 125 small species and 25 large species. 

# Lesson: do not use first column of dataframe as index column. it does not fill the removed duplicate row by 
# successive row. better remove that column. 

# saving the file.
check.to_csv('Iris_size.csv', encoding='utf-8')
# find where is it saved
os.getcwd() #  get current working directory.
 
######### End of the Script. ###########
