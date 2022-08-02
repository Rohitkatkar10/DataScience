# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 20:08:05 2022

@author: rohit
"""

# Logistic Regression Assignment on Election data

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

data = pd.read_csv(r'D:\360digitmg\lectures\25 Logistic Regression\Assignment\Datasets_LR\election_data.csv')
data1 = data.copy()
data.columns

data.drop(0, axis=0, inplace=True)
data.reset_index(inplace=True)

data['Election-id'].nunique() # all 10are unique values 
# Result = target variable
# year = Assuming age of politician

col = ['index','Election-id','year']
data.drop(col, axis=1, inplace=True)

#EDA 
data.isna().sum()
data.duplicated().sum()
data.shape # (10, 4)
data.info()

# change columns for easy typing
data.columns = 'Result','Year', 'Amount_Spent','Popularity_Rank'

# model Building 
import statsmodels.formula.api as smf 
f = '{}'.format('+'.join(data.columns))
ml1 = smf.logit('Result ~ Amount_Spent + Popularity_Rank', data=data).fit()

# logit used for logistic regression.
ml1.summary()
ml1.summary2() # AIC = 9.81

# prediction and getting probabilities
pred = ml1.predict(data.iloc[:, data.columns != 'Result'])

# getting FPR, TPR and threshold
from sklearn.metrics import roc_curve, auc, classification_report
fpr, tpr, thresholds = roc_curve( data.Result, pred)

# finding optimal threshold
optimal_idx = np.argmax(tpr-fpr) # getting index of max difference

optimal_threshold = thresholds[optimal_idx]
optimal_threshold # 0.6793594455096578

# plot roc curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show() # covering most of the area.

# getting numrical value of AUC
roc_auc = auc(fpr, tpr)
roc_auc # 0.9583333333333334

# Create column of all zeros
data['pred_col'] = np.zeros(10)

# coverting prob in pred into 0 and 1 using optimal_threshold
data.loc[pred>optimal_threshold, 'pred_col']=1

# getting classification report
classificaiton = classification_report(data.Result, data.pred_col)

# every thing is fine. we have optimal threshold.
#### splitting the data
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size = 0.2) # 20 %test data

# model Building 
model = smf.logit('Result ~ Amount_Spent + Popularity_Rank', data=train_data).fit()
model.summary() 
model.summary2() # AIC = 9.81

# getting test data probabilities
test_prob = model.predict(test_data)

# getting FPR, TPR and threshold for test data
fpr1, tpr1, thresholds1 = roc_curve( test_data.Result, test_prob)

# ROC plot for test data
plt.plot(fpr1, tpr1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show() # covering most of the area.

test_auc = auc(fpr1, tpr1)
print(test_auc) # 100%

# getting classification Report
# convert test_prob into binary
test_data['test_pred_col'] = np.zeros(2)

test_data.loc[test_prob > optimal_threshold, 'test_pred_col']=1

classification_test = classification_report(test_data.Result, test_data.test_pred_col)
# test data giving excellent accuracy.

# training data accuracy check
train_prob = model.predict(train_data)

# getting FPR, TPR and threshold for train data
fpr2, tpr2, thresholds3 = roc_curve( train_data.Result, train_prob)

# ROC plot for train data
plt.plot(fpr2, tpr2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show() # covering most of the area.

train_auc = auc(fpr2, tpr2)
print(train_auc) # 93.33%

# creating new column to save predictions
train_data['train_pred_col'] = np.zeros(8)

# covert to binaray from train probabilities
train_data.loc[train_prob > optimal_threshold, 'train_pred_col'] = 1

# classification report for train data
classification_train = classification_report(train_data.Result, train_data.train_pred_col)

# get confusion matrix for both datasets
confusion_test = pd.crosstab(test_data.Result, test_data.test_pred_col )
confusion_test


confusion_train = pd.crosstab(train_data.train_pred_col, train_data.Result )
confusion_train

# accuracies
accuracy_test = (1+1)/2
accuracy_train = (3+4)/8
# Can see these accuracies in classification too.

print('The training set accuracy is {}% and testing set accuracy is {}%'.format(round(accuracy_train*100,2), round(accuracy_test*100,2)))


# End of the script