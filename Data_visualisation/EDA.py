# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 20:13:26 2022

@author: rohit
"""

# set working directory
import os
os.chdir(r'D:\360digitmg\lectures\4 statistical data visualization plot')

# load dataset
import pandas as pd # for data manupulation
data1 = pd.read_csv('Datasets/Q1_a.csv') # speed and distance
data2 = pd.read_csv('Datasets/Q2_b.csv') # speed and Weight

data1.columns # index columns is there
data2.columns # unnamed column is there

# remove these two columns
data1.drop('Index', axis=1, inplace=True)
data2.drop('Unnamed: 0', axis=1, inplace=True)


# import library for visualization 
import seaborn as sns


######################## For Data1 ( speed, dist) ############################

# 3rd and 4th business decisions.

# skewness = 0 : normally distributed.
# skewness > 0 : more weight in the left tail of the distribution (right skew).
# skewness < 0 : more weight in the right tail of the distribution (left skew). 

data1.speed.skew()  # -0.11750986144663393
data1.dist.skew()   # 0.8068949601674215
# speed variable has negative skewness value, hence this variable is left skewed or negative skewness.
# dist variable has positive skewness value, hence this variable is right skewed or positive skewness.
# hence both the variables are not perfect normally distributed. 

# for mean mode median
sns.histplot(x=data1.speed) # left skew
# from graph (subjective values), median ~ 12, mode ~ 11 (of speed 13 to 16), mean = Near to 10,
# mean < median.

sns.histplot(x=data1.dist)# right skew
# from graph (subjective values), median ~ 60, mode ~ 13 (of dist 40 ), mean = between 60-80,
# mean > median.

# for variance and std deviation.
import pylab as plt

plt.plot(data1.speed)
# less variance and deviation from mean 
plt.plot(data1.dist)
# more varince and std deviation compared to speed variable.


##### kurtosis
data1.speed.kurt()  # -0.5089944204057617
data1.dist.kurt()   # 0.4050525816795765

# For excess kurtosis 
# = 0 or ~ 0  : Mesokurtic
# > 0 (+ve)   : Leptokurtic
# < 0 (-ve)   : platykurtic

# to calculate excess kurtosis
from scipy.stats import kurtosis
kurtosis(data1.speed)
kurtosis(data1) # for speed and dist = array([-0.57714742,  0.24801866])

# for speed variable kurtosis value is -ve, distribution is platykurtic (has less outliers)
# for dist variable kurtosis value is +ve, distribution is leptokurtic (has more outliers).




############### For data2 ( SP, WT) ##############

# 3rd and 4th business decisions.

# skewness = 0 : normally distributed.
# skewness > 0 : more weight in the left tail of the distribution (right skew).
# skewness < 0 : more weight in the right tail of the distribution (left skew). 

data2.SP.skew()     # 1.6114501961773586
data2.WT.skew()     # -0.6147533255357768
# SP variable has positive skewness value, hence this variable is right skewed or positive skewness.
# WT variable has Negative skewness value, hence this variable is Left skewed or Negative skewness.
# hence both the variables are not normally distributed. 


### histrogram
sns.histplot(x=data2.SP) # right skew
# mean ~ 140, mode ~ 23-24 (of 115 - 120), median = 135
# mean > median

sns.histplot(x=data2.WT)# left skew approximately
# from graph (subjective values), median ~ 35, mode ~ 23 (of WT 30 ), mean =~  30,
# mean ~< median.

##### kurtosis
data2.SP.kurt()  # 2.9773289437871835
data2.WT.kurt()   # 0.9502914910300326

# For excess kurtosis 
# = 0 or ~ 0  : Mesokurtic
# > 0 (+ve)   : Leptokurtic
# < 0 (-ve)   : platykurtic

# to calculate excess kurtosis
from scipy.stats import kurtosis
kurtosis(data2.SP)
kurtosis(data2) # for SP and WT = array([2.72352149, 0.81946588])

# for SP variable kurtosis value is +ve, distribution is leptokurtic (has more outliers) 
# for WT variable kurtosis value is ~ zero, distribution is Mesokrtic (almost equal to Normal Distribution)

########## END 