# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:38:32 2022

@author: rohit
"""

# Assignment on Naive Bayes theorem on Car purchase data.

import pandas as pd
import numpy as np

# load the data
car = pd.read_csv(r'D:\360digitmg\lectures\16 ML Classifier Technique-Naive Bayes\Assignment\dataset\NB_Car_Ad.csv')

# make a copy of data
car2=car.copy()

#### Data Preprocessing ####
# check data info
car.info()
car.shape  # dimension 400X5

# check business decisions
stats = car.describe()  # Numeric data is not in same scale. we may need to scale the data.

# check duplicates 
car.duplicated().sum() # NO duplicates

# Check null values
car.isna().sum() # No null values

# user id is nominal data remove it.
car.drop( 'User ID', axis=1, inplace=True)

car.columns
# Here purchased in label variable, rest are features.



# see outliers in features
import matplotlib.pyplot as plt

plt.boxplot(car['Age'], whis=3) # No outliers

plt.boxplot(car['EstimatedSalary'], whis=3)  # No outliers

# check unuque values in features.
for column in car.columns:
    print(column, '=', car[column].nunique(), 'and', car[column].unique())
    print(' ')
# No abnormal value all values are either numeric or descrete. 

# Declare the features and label.
X = car.drop('Purchased', axis=1)
y = car.Purchased

# Now split the data into training ans testing set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# gender is cotegorical column
import category_encoders as ce
encoder = ce.OneHotEncoder(cols = ['Gender'])

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# scale  the numeric data.
cols = X_train.columns 

from sklearn.preprocessing import RobustScaler 
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test) # both are arrays and column names are numbers.

# make DataFrame and give column names
X_train = pd.DataFrame(X_train, columns = [cols])
X_test = pd.DataFrame(x_test, columns = [cols])


# model training 
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()

# fit the data
bnb.fit(X_train, y_train)

# predict the results 
y_pred_test = bnb.predict(X_test)

# get the accuracy 
from sklearn.metrics import accuracy_score
test_accuracy = accuracy_score(y_test, y_pred_test)
print(test_accuracy*100)

# get training set accuracy for overfitting of model.
y_pred_train = bnb.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(train_accuracy*100)

# get both accuracies
print('Training set accuracy:',train_accuracy*100 )
print('Testing set accuracy:',test_accuracy*100 )
# underfitting model. difference in two is 7%.

# confusion matrix
from sklearn.metrics import confusion_matrix 
confusion_matrix( y_test, y_pred_test)


# alpha = 2
bnb = BernoulliNB(alpha=0.25)

# fit the data
bnb.fit(X_train, y_train)

# predict the results 
y_pred_test = bnb.predict(X_test)

# get the accuracy 
from sklearn.metrics import accuracy_score
test_accuracy = accuracy_score(y_test, y_pred_test)
print(test_accuracy*100)


# get training set accuracy for overfitting of model.
y_pred_train = bnb.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(train_accuracy*100)

# Traied various combination of alpha valeus but no change in accuracies.
print('Training set accuracy:',train_accuracy*100 )
print('Testing set accuracy:',test_accuracy*100 )


#### end of the script ####
