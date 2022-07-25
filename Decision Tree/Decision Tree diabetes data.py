# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 20:57:30 2022

@author: rohit
"""

# Assignment: Decision Tree on Diabetes data.

import pandas as pd
import numpy as np

dia = pd.read_csv(r'D:\360digitmg\lectures\18 Decision Tree\Assignment\Datasets_DTRF\Diabetes.csv')
dia.columns 

dia.info()
# change name of class variable to 'outcome' and rename all the column, since there is space before their names.
for col in dia.columns:
    dia.rename(columns={col:col[1:]}, inplace=True)
dia.rename(columns={'Class variable':'outcome'}, inplace=True)   
dia.columns     
    
# preprocessing 
dia.isnull().sum()  # No null
dia.duplicated().sum() # no duplicates

# check variance 
dia.iloc[:, :8].var(axis=0) == 0  # no zero variance

# Declare the input and output variables.
col = list(dia.columns)
features = col[:8]
label = col[8] 

# split the data
from sklearn.model_selection import train_test_split    
train,test = train_test_split(dia, test_size=0.2) 

# Model Training 
from sklearn.tree import DecisionTreeClassifier     
dtc = DecisionTreeClassifier(criterion='entropy')

# Data fitting
dtc.fit(train[features], train[label])

# prediction
pred_test = dtc.predict(test[features])    
accuracy_test = round(np.mean(test[label] == pred_test)*100,2)
pd.crosstab(test[label], pred_test, rownames=['Actual'], colnames=['Predictions'])  

# prediciton on train data
pred_train = dtc.predict(train[features])
accuracy_trian = round(np.mean(train[label]==pred_train)*100,2)
pd.crosstab(train[label], pred_train, rownames=['Actual'], colnames=['Predictions'])  

# Overfitting.

# try with different parameters.

# Model Training 
from sklearn.tree import DecisionTreeClassifier     
dtc = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)

# Data fitting
dtc.fit(train[features], train[label])

# prediction
pred_test = dtc.predict(test[features])    
accuracy_test1 = round(np.mean(test[label] == pred_test)*100,2)
pd.crosstab(test[label], pred_test, rownames=['Actual'], colnames=['Predictions'])  

# prediciton on train data
pred_train = dtc.predict(train[features])
accuracy_trian1 = round(np.mean(train[label]==pred_train)*100,2)
pd.crosstab(train[label], pred_train, rownames=['Actual'], colnames=['Predictions'])  


# Model Training 
from sklearn.tree import DecisionTreeClassifier     
dtc = DecisionTreeClassifier(criterion='entropy', min_samples_split=3)

# Data fitting
dtc.fit(train[features], train[label])

# prediction
pred_test = dtc.predict(test[features])    
accuracy_test2 = round(np.mean(test[label] == pred_test)*100,2)
pd.crosstab(test[label], pred_test, rownames=['Actual'], colnames=['Predictions'])  

# prediciton on train data
pred_train = dtc.predict(train[features])
accuracy_trian2 = round(np.mean(train[label]==pred_train)*100,2)
pd.crosstab(train[label], pred_train, rownames=['Actual'], colnames=['Predictions'])  

# Model Training 
from sklearn.tree import DecisionTreeClassifier     
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=4)

# Data fitting
dtc.fit(train[features], train[label])

# prediction
pred_test = dtc.predict(test[features])    
accuracy_test3 = round(np.mean(test[label] == pred_test)*100,2)
pd.crosstab(test[label], pred_test, rownames=['Actual'], colnames=['Predictions'])  

# prediciton on train data
pred_train = dtc.predict(train[features])
accuracy_trian3 = round(np.mean(train[label]==pred_train)*100,2)
pd.crosstab(train[label], pred_train, rownames=['Actual'], colnames=['Predictions'])  

# All parameter and their accuracies

All= {'Parameter':["criterion='entropy'", "criterion='entropy', min_samples_leaf=3","criterion='entropy', min_samples_split=3","criterion='entropy', max_depth=4"],
      'Training accuracy %':[accuracy_trian,accuracy_trian1,accuracy_trian2,accuracy_trian3],
      'Testing Accuracy %':[accuracy_test,accuracy_test1,accuracy_test2,accuracy_test3]}

all_accuracies = (pd.DataFrame(All))
print(all_accuracies)


# Fourth model with parameter criterion='entropy'and  max_depth=4  is giving good testing accuracy = 72.08 % compared to others.

########## END of the script ########