# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:51:40 2022

@author: rohit
"""
'''
Name - Rohit katkar
Batch ID- 23122021

'''

##############  OUTLIERS Assigenment    ###########################

# First we will import important and required libraries in one go.

import os # for working directory.
import pandas as pd # for Data manipulation.
import numpy as np # for basic numeric operation
import matplotlib.pyplot as plt # for graphical representation.
import seaborn as sns # advanced visualization.

# import matplotlib.pylab as plt # pylab is combination of pyplot and numpy, used for visualization.

# set working directory. will select directly of dataset folder. 
os.chdir(r'D:\360digitmg\lectures\3 Data preprocessing\DataSets')

# now load the dataset file boston_data.csv to spyder.
data = pd.read_csv('boston_data.csv')

# check dimension, data types and null values.
data.shape # its 404*14

data.info() # No null values and all float values


# check statistics of data.
data.describe()
# since all data is not visible, save to variable

stats=data.describe()
'''
most of the variables have extreme maximum values as compared to their median (50% quantile)
hence, we can say that, there might have outliers. also some of the valriables have median and mean cannot
be corelated with each other.
'''

#check for missing values.
data.isna()
data.isna().sum() # No null values present data set.

# check for duplicate values
data.duplicated().sum() # No Duplicate values.

# check outliers by visualiztion 
data.columns

sns.boxplot(data.crim)    # outliers= yes
sns.boxplot(data.zn)      # outliers= yes
sns.boxplot(data.indus)   # outliers=  NO
sns.boxplot(data.chas)    # outliers= yes
sns.boxplot(data.nox)     # outliers=  No
sns.boxplot(data.rm)      # outliers= yes 
sns.boxplot(data.age)     # outliers=  No
sns.boxplot(data.dis)     # outliers= yes
sns.boxplot(data.rad)     # outliers=  No
sns.boxplot(data.tax)     # outliers=  NO
sns.boxplot(data.ptratio) # outliers= yes
sns.boxplot(data.lstat)   # outliers= yes
sns.boxplot(data.medv)    # outliers= yes
sns.boxplot(data.black)   # outliers= yes    

'''
out of 14 variables, 9 variables have outliers as per box plot.
'''

# second method= outliers by IQR method  (just for example)
IQR = data['crim'].quantile(0.75)-data['crim'].quantile(0.25)
print(IQR)
upper_limit = data['crim'].quantile(0.75) + (1.5*IQR)
lower_limit = data['crim'].quantile(0.25) - (1.5*IQR) # -ve value. since we cannot take values for analysis, consider it zero.
lower_limit=0

'''
In this we can calculate IQR, Up_limit and lower_limit for all outlier variables.
since removing outliers will result in loss of data. we will replace the outlier values with upper and lower limits.
1. by manual method,
2. by function (wisorizer)

'''

'''First will try to create empty DataFrame and will fill outliers with upper and lower limit. 
will append it to new dataframe.
just trying, see if it gets successful.
'''
empty_df = pd.DataFrame()

empty_df['gd_crim']=(np.where(data['crim']>upper_limit, upper_limit, np.where(data['crim']< lower_limit, lower_limit, data['crim'])))
sns.boxplot(empty_df.gd_crim)  # NO outliers.
sns.boxplot(data.crim)          # outliers= yes  

#  directly running above command to see automatically create Dataframe and append trated crim column.

empty_df1['treated_crim'] = (np.where(data['crim']>upper_limit, upper_limit, np.where(data['crim']< lower_limit, lower_limit, data['crim']))) # not successful

# try again with pd.DataFrame fuction.
empty_df1['treated_crim'] = pd.DataFrame(np.where(data['crim']>upper_limit, upper_limit, np.where(data['crim']< lower_limit, lower_limit, data['crim']))) # not successful


# we will use winsorizer function to treat the outliers and save them in new vaariable.
# will try using each capping method, keeping rest of the paremeters same for every method.

from feature_engine.outliers import Winsorizer

winsor = Winsorizer(capping_method='iqr',tail='both', fold=1.5, variables=['crim','zn','chas','rm','dis','ptratio','lstat','medv','black'])

# fit the finction and transform
Treated_iqr_boston=winsor.fit_transform(data[['crim','zn','chas','rm','dis','ptratio','lstat','medv','black']])

sns.boxplot(Treated_iqr_boston.crim)    # outliers= yes
sns.boxplot(Treated_iqr_boston.zn)      # outliers= yes
sns.boxplot(Treated_iqr_boston.indus)   # outliers= NO
sns.boxplot(Treated_iqr_boston.chas)    # outliers= yes
sns.boxplot(Treated_iqr_boston.nox)     # outliers= No
sns.boxplot(Treated_iqr_boston.rm)      # outliers= yes 
sns.boxplot(Treated_iqr_boston.age)     # outliers= No
sns.boxplot(Treated_iqr_boston.dis)     # outliers= yes
sns.boxplot(Treated_iqr_boston.rad)     # outliers= No
sns.boxplot(Treated_iqr_boston.tax)     # outliers= NO
sns.boxplot(Treated_iqr_boston.ptratio) # outliers= yes
sns.boxplot(Treated_iqr_boston.lstat)   # outliers= yes
sns.boxplot(Treated_iqr_boston.medv)    # outliers= yes   

# all outliers are removed.
Treated_iqr_boston.shape  #404x8
data.shape                #404x14

# need to add Non-Outliers column to treated data.
Treated_iqr_boston['indus'] = data['indus']
Treated_iqr_boston['nox'] = data['nox']
Treated_iqr_boston['age'] = data['age']
Treated_iqr_boston['rad'] = data['rad']
Treated_iqr_boston['tax'] = data['tax']

Treated_iqr_boston.shape  #404x14

data.columns


# check for varience 

Treated_iqr_boston.var()
Treated_iqr_boston.info()

treated_stat=Treated_iqr_boston.describe()


'''
variables like 'rm','chas','nox' have zero or near to zero variance, therefore 
we cannot exctract meaning information from them.

variables like 'tax','age','black','zn' have more variance, hence we can find out more 
information from them.
'''
# import these file to computer system.

Treated_iqr_boston.to_csv('trated_boston_data.csv',encoding='utf-8')

# find where is it saved
os.getcwd() #  get current working directory.


