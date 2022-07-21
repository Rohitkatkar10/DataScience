# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 18:25:24 2022

@author: rohit
"""

# Assignment: K-Nearest Neighbor on Glass data.

import pandas as pd
import numpy as np

glass = pd.read_csv(r"D:\360digitmg\lectures\17 ML KNN\Assignments\Datasets_KNN\glass.csv")

glass.columns 

# changing column names to full forms 
cols = ['rhodium', 'sodium','magnesium','aluminium','silicon','potassium','caesium','barium','iron','Type']

### preprocessing
glass.info()
stats = glass.describe()
glass.Type.value_counts()

glass.isna().sum() # no
glass.duplicated().sum() # one duplicate
glass.drop_duplicates(inplace=True, keep='first')

# Need to change dtype of TYPE column, since we classify data for glass type.
glass.Type = glass.Type.astype('O')

# Need to check outliers in the numeric data.
import matplotlib.pyplot as plt 
glass.columns 
plt.boxplot(glass['RI'])
plt.boxplot(glass['Na'])
plt.boxplot(glass['Mg']) # NO
plt.boxplot(glass['Al'])
plt.boxplot(glass['Si'])
plt.boxplot(glass['K'])
plt.boxplot(glass['Ca'])
plt.boxplot(glass['Ba'])
plt.boxplot(glass['Fe'])

# Except variable 'Mg' evry column has outliers but I think all the combinations are composition to make glass,
# on basis glass is categorised. Hence i assume all the outlier values are true. 


# Normilization (I don't think it is necessary, but lets do it)
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalize data except Type column
glass_norm = norm_func(glass.drop('Type',axis=1))
stats_norm = glass_norm.describe() # data successfully normalized

# declare features and label
X = glass_norm
y = pd.DataFrame(glass.Type)

# Data split 
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Model training 
from sklearn.neighbors import KNeighborsClassifier  # when output is discrete data.

# from sklearn.neighbors import KNeighborsRegressor  # when output is continuous data.
# both are same but used for different outputs.

# create instance
knnc = KNeighborsClassifier(n_neighbors=21)  # n_neighbors is K value. selected randomly.
knnc.fit(X_train, y_train)

# wrong label data type
y_train.info() # series has no data types
# convert to dataframe
y_train = pd.DataFrame(y_train)
y_train.info() # object data type

# no try again 
knnc.fit(X_train, y_train)

# change dtype of label to integer.
y_train.Type = y_train['Type'].astype('int')

# no try again 
knnc.fit(X_train, y_train) # worked

# now change y_test to integer
y_test.info()
y_test.Type = y_test['Type'].astype('int')

# predict
pred = knnc.predict(X_test)

# accuracy 
accuracy_test = np.mean(y_test ==pred)

y_test = np.array(y_test)
# accuracy 
accuracy_test = np.mean(y_test ==pred)
print(accuracy_test)
# pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# Training data prediction 
train_pred = knnc.predict(X_train)
# accuracy 
y_train = np.array(y_train)
accuracy_train = np.mean(y_train ==train_pred)
# pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 

print('The Training & Testing data accuracies: {},{}'.format( round(accuracy_train*100, 2), round(accuracy_test*100, 2)))
# clearly, model is overfitting. 


# To get optimum K value
# creating empty list variable 
accuracies = []
# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

 for i in range(3,50,2):
     knnc = KNeighborsClassifier(n_neighbors=i)
     knnc.fit(X_train, y_train)
     train_acc = np.mean(knnc.predict(X_train) == y_train)
     test_acc = np.mean(knnc.predict(X_test) == y_test)
     accuracies.append([train_acc, test_acc])

# Error curve
import matplotlib.pyplot as plt # library to do visualizations 

plt.plot(np.arange(3,50,2),[i[0] for i in accuracies],"ro-",label="train")
plt.plot(np.arange(3,50,2),[i[1] for i in accuracies],"bo-",label="test")
plt.xlabel('k Value')
plt.ylabel('Error')
plt.legend()

# K values = 45. 

##### ENd of the script ####







