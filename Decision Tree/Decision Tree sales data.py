# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Assignment: Decision Tree on company sales data.

import pandas as pd
import numpy as np

sales = pd.read_csv(r'D:\360digitmg\lectures\18 Decision Tree\Assignment\Datasets_DTRF\Company_Data.csv')

sales.columns 

# Preprocessing 
sales.info()

sales.isnull().sum() # no null
sales.duplicated().sum() # no

stats = sales.describe()

sales.ShelveLoc.value_counts()
sales.Urban.value_counts()
sales.US.value_counts()

cat_col = [w for w in sales.columns if sales[w].dtype == 'O']

sales.Sales.value_counts()
    
# convert Sales into categorical data. divide it into High, Medium and Low sales.
# Create new column for it. and remove sales column
sales['sales_cat'] = pd.DataFrame(np.where(sales.Sales>10, 'High', np.where(sales.Sales<=5, 'Low', 'Medium')))
sales.drop('Sales', axis=1, inplace=True)

# converting categorical data into labelencoding
from sklearn.preprocessing import LabelEncoder   

LE = LabelEncoder()    

for col in cat_col:
    sales[col] = LE.fit_transform(sales[col])

# Declare features and label column names
colNames = list(sales.columns)
# last column is output here.
features = colNames[:10]
label = colNames[10] 

# now split the data into training and testing set
from sklearn.model_selection import train_test_split 

train, test = train_test_split(sales, test_size = 0.2)

# Model training 
from sklearn.tree import DecisionTreeClassifier 
dt = DecisionTreeClassifier(criterion='entropy')

dt.fit(train[features], train[label])

# prediction 
pred_test = dt.predict(test[features])

# Evaluation
test_accuracy = np.mean(test[label] == pred_test)
pd.crosstab(test[label], pred_test, rownames=['Actual'], colnames=['prediction'])
print(test_accuracy)

# training data prediction 
pred_train = dt.predict(train[features])    
train_accuracy = np.mean(train[label] == pred_train)
pd.crosstab(train[label], pred_train, rownames=['Actual'], colnames=['predictions'])
print('The training set accuracy is {} % and testing set accuracy is {} %'.format(round(train_accuracy*100,2), round(test_accuracy*100,2)))

# Notes: The Decision Tree model always tends to overfit. we can solve this by pruning the model.
# since it is not mentioned in quenstion which parameter to use for pruning, I leave model as is.


#### ENd of the script ###


# just trying various combination for pruning. But all below of them are not giving satisfactory result.
# will stick to above model. 

# Model training 
from sklearn.tree import DecisionTreeClassifier 
dt = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)

dt.fit(train[features], train[label])

# prediction 
pred_test = dt.predict(test[features])

# Evaluation
test_accuracy = np.mean(test[label] == pred_test)
pd.crosstab(test[label], pred_test, rownames=['Actual'], colnames=['prediction'])
print(test_accuracy)

# training data prediction 
pred_train = dt.predict(train[features])    
train_accuracy = np.mean(train[label] == pred_train)
pd.crosstab(train[label], pred_train, rownames=['Actual'], colnames=['predictions'])
print('The training set accuracy is {} % and testing set accuracy is {} %'.format(round(train_accuracy*100,2), round(test_accuracy*100,2)))

######################################################################################
######################################################################################

# Model training 
from sklearn.tree import DecisionTreeClassifier 
dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=3)

dt.fit(train[features], train[label])

# prediction 
pred_test = dt.predict(test[features])

# Evaluation
test_accuracy = np.mean(test[label] == pred_test)
pd.crosstab(test[label], pred_test, rownames=['Actual'], colnames=['prediction'])
print(test_accuracy)

# training data prediction 
pred_train = dt.predict(train[features])    
train_accuracy = np.mean(train[label] == pred_train)
pd.crosstab(train[label], pred_train, rownames=['Actual'], colnames=['predictions'])
print('The training set accuracy is {} % and testing set accuracy is {} %'.format(round(train_accuracy*100,2), round(test_accuracy*100,2)))

######################################################################################
######################################################################################

# Model training 
from sklearn.tree import DecisionTreeClassifier 
dt = DecisionTreeClassifier(criterion='entropy', max_depth=4)

dt.fit(train[features], train[label])

# prediction 
pred_test = dt.predict(test[features])

# Evaluation
test_accuracy = np.mean(test[label] == pred_test)
pd.crosstab(test[label], pred_test, rownames=['Actual'], colnames=['prediction'])
print(test_accuracy)

# training data prediction 
pred_train = dt.predict(train[features])    
train_accuracy = np.mean(train[label] == pred_train)
pd.crosstab(train[label], pred_train, rownames=['Actual'], colnames=['predictions'])
print('The training set accuracy is {} % and testing set accuracy is {} %'.format(round(train_accuracy*100,2), round(test_accuracy*100,2)))

######################################################################################
######################################################################################

