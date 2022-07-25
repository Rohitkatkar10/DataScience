# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 14:28:38 2022

@author: rohit
"""


# Assignment: Decision Tree on Income data.

import pandas as pd
import numpy as np

income = pd.read_csv(r'D:\360digitmg\lectures\18 Decision Tree\Assignment\Datasets_DTRF\HR_DT.csv')
income.columns 

income.info()
   
# preprocessing 
income.isnull().sum()  # No null
income.duplicated().sum() # yes duplicates
income.drop_duplicates(inplace=True, keep='first')


# check variance 
income.var(axis=0) == 0  # no zero variance


# get value count for discrete data
cat_col = [w for w in income.columns if income[w].dtype=='O']
num_col = [w for w in income.columns if w not in cat_col]
stats_cat = income[cat_col].describe()
stats_num = income.describe()

# label encoding 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in cat_col:
    income[col] = le.fit_transform(income[col])


# Declare input and output
X = income.iloc[:, [0,1]]
y = income.iloc[:, 2]


# split the data
from sklearn.model_selection import train_test_split    
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Model Training 
from sklearn.tree import DecisionTreeRegressor     
regtree = DecisionTreeRegressor(max_depth = 3)

# max_depth = takes 3 internal node levels in tree from root node. 

regtree.fit(x_train, y_train)

# Prediction
test_pred = regtree.predict(x_test)
train_pred = regtree.predict(x_train)

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score

# Error on test dataset
mean_squared_error(y_test, test_pred)
r2_score(y_test, test_pred)

# Error on train dataset
mean_squared_error(y_train, train_pred)
r2_score(y_train, train_pred)


# Plot the DT
#dot_data = tree.export_graphviz(regtree, out_file=None)
#from IPython.display import Image
#import pydotplus
#graph = pydotplus.graph_from_dot_data(dot_data)
#Image(graph.create_png())


# Pruning the Tree
# Minimum observations at the internal node approach
regtree2 = DecisionTreeRegressor(min_samples_split = 3)
# Min_sample_split = here at each internal node, to split further there 
# should at least three records to split further if less than 3 then 
# it will not split at that intenal node.

# or The minimum number of samples required to split an internal node:
regtree2.fit(x_train, y_train)

# Prediction
test_pred2 = regtree2.predict(x_test)
train_pred2 = regtree2.predict(x_train)

# Error on test dataset
mean_squared_error(y_test, test_pred2)
r2_score(y_test, test_pred2)

# Error on train dataset
mean_squared_error(y_train, train_pred2)
r2_score(y_train, train_pred2)

###########
## Minimum observations at the leaf node approach
regtree3 = DecisionTreeRegressor(min_samples_leaf = 3)
# The minimum number of samples required to be at a leaf node.
# A split point at any depth will only be considered if it leaves
# at least min_samples_leaf training samples in each of the left and right branches. 
# This may have the effect of smoothing the model, especially in regression.

regtree3.fit(x_train, y_train)

# Prediction
test_pred3 = regtree3.predict(x_test)
train_pred3 = regtree3.predict(x_train)

# measure of error on test dataset
mean_squared_error(y_test, test_pred3)
r2_score(y_test, test_pred3)

# measure of error on train dataset
mean_squared_error(y_train, train_pred3)
r2_score(y_train, train_pred3)


# last model suits best 

############ End of the script #############