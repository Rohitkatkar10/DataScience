# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:02:04 2022

@author: rohit
"""

# Assignment on Ensemble Technique on Tumor_Ensemble data.

import pandas as pd 
import numpy as np

data = pd.read_csv(r"D:\360digitmg\lectures\19 Ensemble Techniques\Assignment\Datasets\Tumor_Ensemble.csv")

data.columns 

data.info()
data.drop('id', axis=1, inplace=True)
# we are mapping the diagnosis column values in binary. B=0 M=1
out = pd.get_dummies(data["diagnosis"], drop_first='Yes')
out = np.array(out)
data['diagnosis'] = out

# First preprocess the data
data.isnull().sum()  # no null
data.duplicated().sum()  # no duplicates 

data.var(axis=0) == 0 # no zero variance

# declare features and labels 
features = data.iloc[:, data.columns != 'diagnosis']
label = data['diagnosis']   
    
# split the data 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state= 42 )

################## Voting ################################
# import 3 base learner and ensemble voting classifier
from sklearn import linear_model, svm, neighbors, naive_bayes
from sklearn.ensemble import VotingClassifier 

# Creating instances for base learners
l1 = neighbors.KNeighborsClassifier(n_neighbors=5)
l2 = linear_model.Perceptron(tol=1e-2, random_state=0)
l3 = svm.SVC(gamma=0.001)

# Creating instance for voting classifier
voting = VotingClassifier([('KNN', l1),('Prc',l2),('SVM',l3)]) # voting = 'hard' default.

# fit the training dat
voting.fit(X_train, y_train)

# prediciton 
hard_prediction = voting.predict(X_test)

# Evaluation
from sklearn.metrics import accuracy_score
Accuracy = accuracy_score(y_test, hard_prediction)
print('Accuracy by Hard voting is {}.'.format(round(Accuracy*100,2)))


# Soft Voting # 
# Instantiate the learners (classifiers)
l4 = neighbors.KNeighborsClassifier(n_neighbors = 5)
l5 = naive_bayes.GaussianNB() # we use thi model when output is given.
l6 = svm.SVC(gamma = 0.001, probability = True)

# prediction using individual models. we can do this also.
l4.fit(X_train, y_train)
l5.fit(X_train, y_train)
l6.fit(X_train, y_train)

# Instantiate the voting classifier
voting = VotingClassifier([('KNN', l4),
                           ('NB', l5),
                           ('SVM', l6)],
                            voting = 'soft')

# Get the base learner predictions
predictions_4 = l4.predict(X_test)
predictions_5 = l5.predict(X_test)
predictions_6 = l6.predict(X_test)

# Accuracies of base learners
print('L4:', accuracy_score(y_test, predictions_4))
print('L5:', accuracy_score(y_test, predictions_5))
print('L6:', accuracy_score(y_test, predictions_6))

# train voting classifier
voting.fit(X_train, y_train)

# prediction 
soft_predictions = voting.predict(X_test)


# Accuracy of Soft voting
Accuracy_soft= accuracy_score(y_test, soft_predictions)
print('Accuracy by doft voting is {}.'.format(round(Accuracy_soft*100,2)))
print('Accuracy by Hard voting is {}.'.format(round(Accuracy*100,2)))
# Both are same


#################  Stacking #######################3
# In stacking, we take many base learners and one meta learner.
# import base learners 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier # Multi-layer percetron classifier. 
# meta learner
from sklearn.linear_model import LogisticRegression

# Create list and add all the learners in it
base_learners = [] # base learners or level 0 models

# KNN classifier model
knn = KNeighborsClassifier(n_neighbors=2)
base_learners.append(knn)

# Decision Tree Classifier model
dtr = DecisionTreeClassifier(max_depth=4, random_state=123456)
base_learners.append(dtr)

# Multi Layered Perceptron classifier
mlpc = MLPClassifier(hidden_layer_sizes =(100, ), solver='lbfgs', random_state=123456)
base_learners.append(mlpc)


# Meta model using Logistic Regression
meta_learner = LogisticRegression(solver='lbfgs') # level 1 model

# lets take predictions from all base learners and save them to list and covert it to DataFrame of dimension 3x614

all_predictions = []
for i in range(len(base_learners)):
    learner = base_learners[i]
    learner.fit(X_train,y_train)
    all_predictions.append(learner.predict(X_train))

meta_data = pd.DataFrame(all_predictions)

# Transpose the meta data to be fed into the meta learner
meta_data = meta_data.transpose() # making dimenison 614X3
# this above data will be traning data for meta learner

# Now create test data for meta data
test_predictions = []

for i in range(len(base_learners)):
    learner = base_learners[i]
    learner.fit(X_train,y_train)
    test_predictions.append(learner.predict(X_test))

test_meta_data = pd.DataFrame(test_predictions)

# Transpose the test meta data to be fed into the meta learner for prediction
test_meta_data = test_meta_data.transpose() # making dimenison 154X3
# this data is test meta data. 

### Now all above two train and test data are the features, now we need label data to train the meta learner.
meta_target = y_train # since we take predictions from base learner as traning and testing data 
# and we take target data for meta learner the same y_train data to train meta learner model.
# we will predict output from test_meta_data using meta_learner and will compare with y_test (which is original output test data)

# Fit the meta learner on the train set and evaluate it on the test set
meta_learner.fit(meta_data, meta_target)
ensemble_predictions = meta_learner.predict(test_meta_data)

from sklearn import metrics # for accuracy_score
acc = metrics.accuracy_score(y_test, ensemble_predictions)
print(round(acc*100,2))


# Print the results
base_acc = []
for i in range(len(base_learners)):
    learner = base_learners[i]
    pred = learner.predict(X_test)
    base_acc.append(round(metrics.accuracy_score(y_test, pred)*100,2))

All_accuracy = pd.DataFrame({'base_learner':base_learners, 'base_accuracies':base_acc})
print(All_accuracy) # base model accuracy
print(round(acc*100,2)) # ansemble accuracy
# Ensebles accuracy is equal to Decision Trees accuracy.

############ Bagging ###########

from sklearn import tree
clftree = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier
# A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets 
# of the original dataset and then aggregate their individual predictions (either by voting or by averaging)
# to form a final prediction.
# see the documentaion for various parameters

bag_clf = BaggingClassifier(base_estimator = clftree, n_estimators = 500,
                            bootstrap = True, n_jobs = 1, random_state = 42)
# default base_estmator is decision Tree classifier. we can ML model other than decisio tree also.
# IN bagging we have to give bas estimator. whearas in Random forest classfier it is by default Decision Trees.

# here in bagging only rows are randomly choosen, number of columns will be same as that of original data.
# But in random forest classifier both rows and columns are randomly choosen.
# n_estimator = 500, 500  samples taken and 500 ML models parallely trained.
# n_jobs = 1, this means The number of jobs to run in parallel for both fit() and predict().
# None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
  
bag_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, bag_clf.predict(X_test))
accuracy_score(y_test, bag_clf.predict(X_test))

# Evaluation on Training Data
confusion_matrix(y_train, bag_clf.predict(X_train))
accuracy_score(y_train, bag_clf.predict(X_train))

# decision tree bagging classifier is giving almost 100% accuracy on both the data



##### Random Forest Classifiers


# since ouput colum is in binary, will go for 'RF classification' instead of 'RF Rregession'.
from sklearn.ensemble import RandomForestClassifier
# A random forest is a meta estimator that fits a number of 
# decision tree classifiers on various sub-samples of the dataset 
# and uses averaging to improve the predictive accuracy and control over-fitting.

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)
# IN bagging we have to give bas estimator. whearas in Random forest classfier it is by default Decision Trees.


# here in bagging only rows are randomly choosen, number of columns will be same as that of original data.
# But in random forest classifier both rows and columns are randomly choosen.
 

rf_clf.fit(X_train, y_train)



confusion_matrix(y_test, rf_clf.predict(X_test))
accuracy_score(y_test, rf_clf.predict(X_test))

# Evaluation on Training Data
confusion_matrix(y_train, rf_clf.predict(X_train))
accuracy_score(y_train, rf_clf.predict(X_train))
# Random Forest classifier is giving almost 100% accuracy on both the data


######
# GridSearchCV (it is the one of the Model Tuning Technique)

from sklearn.model_selection import GridSearchCV

rf_clf_grid = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)
# taking base estimator as random forest, and taking parameters that we wanr keep constant.
# n_estimator =500 to reduce computaion burdon on system.

param_grid = {"max_features": [4, 5, 6, 7, 8, 9, 10], "min_samples_split": [2, 3, 10]}
# here we give different parameters that we want to try. 
# see documentation for max_feature since it take string and numeric values 
# see sir's handon too. for comparison. both codes works fine.
# here we take combination of all the parameters. 4-2, 4-3, 4-10 then 5-2,5-3,5-10 then 4-2,6-3,6-10 and so on
# in this way will take 21 combinaion. hence we will try 21 ML models.


grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')
# n_job = -1. parallel processing 
# n_job = 1. Sequencial processing
# cv = 5 spliting data 5 times.

grid_search.fit(X_train, y_train) # # fitting will take time.

grid_search.best_params_  # will best parameters. by considering constant parameters as is.

cv_rf_clf_grid = grid_search.best_estimator_
# training model using best estimator.


confusion_matrix(y_test, cv_rf_clf_grid.predict(X_test))
accuracy_score(y_test, cv_rf_clf_grid.predict(X_test))

# Evaluation on Training Data
confusion_matrix(y_train, cv_rf_clf_grid.predict(X_train))
accuracy_score(y_train, cv_rf_clf_grid.predict(X_train))

# Random forest classifier is giving almost 100% accuracy on both the data
# still overfitting. we have add  more and different parameters and try again.

###### Boosting ##########

##### 1. AdaBoosting ########
from sklearn.ensemble import AdaBoostClassifier
# An AdaBoost [1] classifier is a meta-estimator that begins by fitting 
# a classifier on the original dataset and then fits additional copies of the classifier 
# on the same dataset but where the weights of incorrectly classified instances are adjusted 
# such that subsequent classifiers focus more on difficult cases.

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)

# learning_rate = Weight applied to each classifier at each boosting iteration.\
#  A higher learning rate increases the contribution of each classifier. \
#  There is a trade-off between the learning_rate and n_estimators parameters.

ada_clf.fit(X_train, y_train)



# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(X_train))
accuracy_score(y_test, ada_clf.predict(X_test))

# Evaluation on Training Data
accuracy_score(y_train, ada_clf.predict(X_train))


# overfitting. 

##### 2. Gradient BOosting

from sklearn.ensemble import GradientBoostingClassifier

boost_clf = GradientBoostingClassifier()

boost_clf.fit(X_train, y_train)


confusion_matrix(y_test, boost_clf.predict(X_test))
accuracy_score(y_test, boost_clf.predict(X_test))


# Hyperparameters
boost_clf2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 1000, max_depth = 1)
boost_clf2.fit(X_train, y_train)


# Evaluation on Testing Data
confusion_matrix(y_test, boost_clf2.predict(X_test))
accuracy_score(y_test, boost_clf2.predict(X_test))

# Evaluation on Training Data
accuracy_score(y_train, boost_clf2.predict(X_train))

# both accuracie are same almost.

# end of the script.