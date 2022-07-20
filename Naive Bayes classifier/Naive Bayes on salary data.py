# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 18:33:10 2022

@author: rohit
"""

# Assignment on Naive Bayes Thereom on Salary data.

import pandas as pd
import numpy as np


# load the data
salary = pd.read_csv(r'D:\360digitmg\lectures\16 ML Classifier Technique-Naive Bayes\Assignment\dataset\SalaryData_Train.csv')

# make a copy of data
salary2=salary.copy()

#### Data Preprocessing ####
# check data info
salary.info()
salary.shape  # dimension 30161x14

# check business decisions
stats = salary.describe()  # Numeric data is not in same scale. we may need to scale the data.

# check duplicates 
salary.duplicated().sum() # 3258 duplicates are there.

# Remove duplicates and confirm
salary.drop_duplicates(inplace=True) # confirmed last command. 

# Check null values
salary.isna().sum() # No null values

salary.shape  # new shape 26903x14

# classify categorical and numeric columns 
cat_col = [var for var in salary.columns if salary[var].dtype == 'O'] # 'O' for Object 
print('There are {} categorical variables'.format(len(cat_col)))
print('The Name of categorical columns are:', cat_col)

num_col = [col for col in salary.columns if col not in cat_col]
print('There are {} numerical columns'.format(len(num_col)))
print('The Name of numerical columns are:', num_col)

# now check outliers in Numerical data. and skewness of the data.
import matplotlib.pyplot as plt
for i in num_col:
    print(i)
    print('plt.boxplot(salary[i])')
    
# copy from console

plt.boxplot(salary['age'], whis=3)

plt.boxplot(salary['educationno'], whis=3) 

plt.boxplot(salary['capitalgain'], whis=3) # yes outliers 

plt.boxplot(salary['capitalloss'], whis=3) # yes outliers 

plt.boxplot(salary['hoursperweek'], whis=3) # yes outliers           
# All last three numeric columns have outliers. since the number of outliers is more.
# if outliers tratement is done on capital gain & loss column then their variance will become zero.
# and also  the number of  outliers is more. it cannot be by mistake.
# hence considering all outliers are as true values 

# check unique values in categorical columns
for i in cat_col:
    print(i,'=', salary[i].unique())
    print("  ")
    
# No abnormal value in cat columns. all good.
# number of unique valeus
for i in cat_col:
    print( i,'=', salary[i].nunique())
  
    
# Declare the feature variables and label varible.
X = salary.drop('Salary', axis=1) # input or independent variables
y = salary.Salary   # outout or Dependent variable.

# split the data into train and test set. we have seperate test data all together 
X_train = salary.drop('Salary', axis=1) # input or independent variables
y_train = salary.Salary   # outout or Dependent variable.

# import test data.
test_data = pd.read_csv(r'D:\360digitmg\lectures\16 ML Classifier Technique-Naive Bayes\Assignment\dataset\SalaryData_Test.csv')
test_data.shape 
salary.shape 
X_train.shape 

# seperate input and output
X_test = test_data.drop('Salary', axis=1) # input or independent variables
y_test = test_data.Salary   # outout or Dependent variable.
X_test.shape 
X_test.shape



# Encode categorical variables.
cat_col = [var for var in X_train.columns if X_train[var].dtype == 'O' ]
num_col = [var for var in X_train if var not in cat_col]
print(cat_col)
print(num_col) 


# encode the categorical data.
import category_encoders as ce 
encoder = ce.OneHotEncoder(cols = cat_col) 

# Here, why .fit_transform for train data and only transform for test data?
X_train = encoder.fit_transform(X_train)    
X_test = encoder.transform(X_test)    

X_train.head()

# make numeric data in one scale
cols = X_train.columns
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Here, in .fit() : fit calculates only necessary parameters in this case i.e. in scaling.
# thses parameters are mean and standard deviation. 
# transform: transform will change all the variables into same scale.
# Z = (x-mean)/(std var) is the formula for scaling.
# fit calculate mean and std var only. but does not calculate the values of Z for each variable. 
# transform takes these to values and calculates these Z values for each variable. and we get data in same scale.

# here parameters mean & std var are different for train & test data sets. 
# since we train model on training set. hence we have to same parameter for test data.

# this is why .fit_transform for train data and only transform for test data
X_train.head() # It is array and has no columns names


X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])


# Now our data is in perfect condition.
# Model Building
from sklearn.naive_bayes import GaussianNB

# create model instance
gnb = GaussianNB()

# fit the model
gnb.fit(X_train, y_train)

# predict the results
y_pred = gnb.predict(X_test)

# calculate accuracy 
accuracy = np.mean(y_pred == y_test) # 80.09%

# another method for accuracy score.
from sklearn.metrics import accuracy_score
accuracy_score(y_pred, y_test)

print('model test accuracy is:', accuracy*100)

# calculate the accuracy of model on trainin set. 
# to see ML is overfitting or underfitting.
y_pred_train = gnb.predict(X_train)

train_accuracy = accuracy_score(y_pred_train, y_train)

print('model train accuracy is:', train_accuracy*100)
print('model test accuracy is:', accuracy*100)

# both accuracies are same. Hence i can say that model is not overfitting.

# see confusion matrix
from sklearn.metrics import confusion_matrix 
confusion_matrix( y_pred , y_test)
# [ TP, FN]
# [ FP, TN]


confusion_matrix( y_test, y_pred)
# this might be columns X rows,
# on columns we have actual values = True and False.
# on rows we have predictions = Positive and negatives. 
# [ TP, FP]
# [ FN, TN]

# here, TP = 9035, TN= 3028, FP=2325, FN=672. 
# there are many FN values, we have to reduce this number. FN is more dangerous than FP. 

# results 
# model train accuracy % is: 80.14323132522131
# model test accuracy % is: 80.09960159362551

# both the accuracies are almost equal, hence the model is not over or underfitting.

############### End of the script #########3