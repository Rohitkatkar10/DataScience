# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:13:02 2022

@author: rohit
"""


### Multinomial Regression ####
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mode = pd.read_csv(r"D:\360digitmg\lectures\26 Multinomial Regression\Assignment\Datasets_Multinomial\mdata.csv")
mode1 = mode.copy()

mode.describe()
mode.prog.value_counts() # target variable

# remove unnecessary columns
col = ['Unnamed: 0','id']
mode.drop(col, axis=1, inplace=True)

mode.columns

# Boxplot of independent variable distribution for each category of prog 
sns.boxplot(x = "prog", y = "read", data = mode)
sns.boxplot(x = "prog", y = "write", data = mode)
sns.boxplot(x = "prog", y = "math", data = mode)
sns.boxplot(x = "prog", y = "science", data = mode)


# Scatter plot for each categorical choice of car
sns.stripplot(x = "prog", y = "science", jitter = True, data = mode)
sns.stripplot(x = "prog", y = "math", jitter = True, data = mode)
sns.stripplot(x = "prog", y = "write", jitter = True, data = mode)
sns.stripplot(x = "prog", y = "read", jitter = True, data = mode)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(mode) # Normal
sns.pairplot(mode, hue = "prog") # With showing the category of each car choice in the scatter plot

# Correlation values between each independent features
mode.corr()
mode.info()


# label encoding
from sklearn.preprocessing import LabelEncoder 
# creating instance for label encoder
labelencoder = LabelEncoder()
X = mode.iloc[:, mode.columns != 'prog']
y = mode.prog

for col in X.columns:
    if X[col].dtype == 'O':
        X[col] = labelencoder.fit_transform(X[col])
        
mode = pd.concat([X, y], axis=1)

# data split
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(X_train, y_train)
help(LogisticRegression)

test_predict = model.predict(X_test) # Test predictions

# Test accuracy 
accuracy_score(y_test, test_predict)

train_predict = model.predict(X_train) # Train predictions 
# Train accuracy 
accuracy_score(y_train, train_predict) 

# End of the script