# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 10:00:46 2022

@author: rohit
"""


# Assignment: Decision Tree on Fraud data.

import pandas as pd
import numpy as np

fraud = pd.read_csv(r'D:\360digitmg\lectures\18 Decision Tree\Assignment\Datasets_DTRF\Fraud_check.csv')
fraud.columns 

fraud.info()
   
# preprocessing 
fraud.isnull().sum()  # No null
fraud.duplicated().sum() # no duplicates

# check variance 
fraud.var(axis=0) == 0  # no zero variance


# get value count for discrete data
cat_col = [w for w in fraud.columns if fraud[w].dtype=='O']
num_col = [w for w in fraud.columns if w not in cat_col]
stats_cat = fraud[cat_col].describe()
stats_num = fraud.describe()

# label encoding 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in cat_col:
    fraud[col] = le.fit_transform(fraud[col])

num_data = fraud.drop(cat_col, axis=1)
cat_data = fraud[cat_col]
num_data_x = num_data.drop('Taxable.Income', axis=1)

# scale the data
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()

data = pd.DataFrame(sc.fit_transform(num_data_x))
data.columns = num_data_x.columns

# convate data label encoding and scaled & # declare  input and output variables
X = pd.concat([cat_data, data], axis=1)
y =(np.where(fraud['Taxable.Income']<=30000, 'Risky', 'Good'))
# y.columns = ['taxable_income']


# split the data
from sklearn.model_selection import train_test_split    
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Model Training 
from sklearn.tree import DecisionTreeClassifier     
dtc = DecisionTreeClassifier(criterion='entropy')

# Data fitting
dtc.fit(X_train, y_train)

# prediction
pred_test =(dtc.predict(X_test))    
accuracy_test = round(np.mean(y_test == pred_test)*100,2)
pd.crosstab(y_test, pred_test, rownames=['Actual'], colnames=['Predictions'])  

# prediciton on train data
pred_train = dtc.predict(X_train)
accuracy_trian = round(np.mean(y_train==pred_train)*100,2)
pd.crosstab(y_train, pred_train, rownames=['Actual'], colnames=['Predictions'])  

# Overfitting.

# try with different parameters.

# Model Training 
from sklearn.tree import DecisionTreeClassifier     
dtc = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)

# Data fitting
dtc.fit(X_train, y_train)

# prediction
pred_test = dtc.predict(X_test)    
accuracy_test1 = round(np.mean(y_test == pred_test)*100,2)
pd.crosstab(y_test, pred_test, rownames=['Actual'], colnames=['Predictions'])  

# prediciton on train data
pred_train = dtc.predict(X_train)
accuracy_trian1 = round(np.mean(y_train==pred_train)*100,2)
pd.crosstab(y_train, pred_train, rownames=['Actual'], colnames=['Predictions'])  


# Model Training 
from sklearn.tree import DecisionTreeClassifier     
dtc = DecisionTreeClassifier(criterion='entropy', min_samples_split=3)

# Data fitting
dtc.fit(X_train, y_train)

# prediction
pred_test = dtc.predict(X_test)    
accuracy_test2 = round(np.mean(y_test == pred_test)*100,2)
pd.crosstab(y_test, pred_test, rownames=['Actual'], colnames=['Predictions'])  

# prediciton on train data
pred_train = dtc.predict(X_train)
accuracy_trian2 = round(np.mean(y_train==pred_train)*100,2)
pd.crosstab(y_train, pred_train, rownames=['Actual'], colnames=['Predictions'])  

# Model Training 
from sklearn.tree import DecisionTreeClassifier     
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=4)

# Data fitting
dtc.fit(X_train, y_train)

# prediction
pred_test = dtc.predict(X_test)    
accuracy_test3 = round(np.mean(y_test == pred_test)*100,2)
pd.crosstab(y_test, pred_test, rownames=['Actual'], colnames=['Predictions'])  

# prediciton on train data
pred_train = dtc.predict(X_train)
accuracy_trian3 = round(np.mean(y_train==pred_train)*100,2)
pd.crosstab(y_train, pred_train, rownames=['Actual'], colnames=['Predictions'])  

# if max_depth goes beyond 4 then test accuracy in decreasing and if max_depth is from 1 to 4  testing accruaracy is same for all.

# All parameter and their accuracies

All= {'Parameter':["criterion='entropy'", "criterion='entropy', min_samples_leaf=3","criterion='entropy', min_samples_split=3","criterion='entropy', max_depth=4"],
      'Training accuracy %':[accuracy_trian,accuracy_trian1,accuracy_trian2,accuracy_trian3],
      'Testing Accuracy %':[accuracy_test,accuracy_test1,accuracy_test2,accuracy_test3]}

all_accuracies = (pd.DataFrame(All))
print(all_accuracies)


# Fourth model with parameter criterion='entropy'and  max_depth=4  is giving good testing accuracy = 75.83 % compared to others.

########## END of the script ########