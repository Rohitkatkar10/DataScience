# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:43:54 2022

@author: rohit
"""

# Assignment: K-Nearest Neighbor on Zoo data.

import pandas as pd
import numpy as np

zoo = pd.read_csv(r"D:\360digitmg\lectures\17 ML KNN\Assignments\Datasets_KNN\Zoo.csv")

zoo.columns 

### preprocessing
zoo.info()
stats = zoo.describe()
animal = zoo['animal name'].value_counts()

zoo.isna().sum() # no
zoo.duplicated().sum() # no duplicates


# since the data itself in range of 0 and 1. no need to scale it.
# Declare label and features
X = np.array(zoo.iloc[:, 1:])
y = np.array(zoo.iloc[:, 0])

# split data into training and testing set

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# train the model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 21)
# randomly taking k value = 21.
knn.fit(X_train, Y_train)

pred_test = knn.predict(X_test)
pred_test

# accuracy 
accuracy_test = np.mean(Y_test == pred_test)
print('The accuracy of testing set is {}.'.format(round(accuracy_test*100, 2)))

# predition with training data
pred_train = knn.predict(X_train)
pred_train

# accuracy on training data
accuracy_train = np.mean(Y_train == pred_train)
print('The accuracy of traing set is {}.'.format(round(accuracy_train*100, 2)))

# both the accuracy are:
print('The accuracy of traing set is {}.'.format(round(accuracy_train*100, 2)))
print('The accuracy of testing set is {}.'.format(round(accuracy_test*100, 2)))

# Model is overfitting here. 
# and also we do not have much observation to train machine learing model.
# there are 100 data points and 99 unique aminals 
# clearly data points are enough to train the model.

#### End of the script #########




