# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:10:28 2022

@author: rohit
"""

# Logistic Regression Assignment on ad data

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

data = pd.read_csv(r'D:\360digitmg\lectures\25 Logistic Regression\Assignment\Datasets_LR\advertising.csv')
data1 = data.copy()
data.columns

data.Ad_Topic_Line.nunique() # all 1000 are unique values 
data.City.nunique()
data.Country.nunique()

# remove categorical data
cat_col = [col for col in data.columns if data[col].dtype == 'O']

data.drop(cat_col, axis=1, inplace=True)

#EDA 
data.isna().sum()
data.duplicated().sum()
data.shape # (1000, 6)

# change columns for easy typing
data.columns = 'time_spent','age', 'area_income','daily_internet_usage', 'Male', 'clicked_on_ad'

# model Building 
import statsmodels.formula.api as smf 
f = '{}'.format('+'.join(data.columns))
ml1 = smf.logit('clicked_on_ad ~ time_spent+age+area_income+daily_internet_usage+Male', data=data).fit()
# logit used for logistic regression.
ml1.summary()
ml1.summary2() # AIC = 193.80

# prediction and getting probabilities
pred = ml1.predict(data.iloc[:, data.columns != 'clicked_on_ad'])

# getting FPR, TPR and threshold
from sklearn.metrics import roc_curve, auc, classification_report
fpr, tpr, thresholds = roc_curve( data.clicked_on_ad, pred)

# finding optimal threshold
optimal_idx = np.argmax(tpr-fpr) # getting index of max difference

optimal_threshold = thresholds[optimal_idx]
optimal_threshold # 0.6326111998951828

# plot roc curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show() # covering most of the area.

# getting numrical value of AUC
roc_auc = auc(fpr, tpr)
roc_auc # 0.9918

# Create column of all zeros
data['pred_col'] = np.zeros(1000)

# coverting prob in pred into 0 and 1 using optimal_threshold
data.loc[pred>optimal_threshold, 'pred_col']=1

# getting classification report
classificaiton = classification_report(data.clicked_on_ad, data.pred_col)

# every thing is fine. we have optimal threshold.
#### splitting the data
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size = 0.2) # 20 %test data

# model Building 
model = smf.logit('clicked_on_ad ~ time_spent+age+area_income+daily_internet_usage+Male', data=train_data).fit()
model.summary() 
model.summary2() # AIC = 173.08

# getting test data probabilities
test_prob = model.predict(test_data)

# getting FPR, TPR and threshold for test data
fpr1, tpr1, thresholds1 = roc_curve( test_data.clicked_on_ad, test_prob)

# ROC plot for test data
plt.plot(fpr1, tpr1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show() # covering most of the area.

test_auc = auc(fpr1, tpr1)
print(test_auc) # 99.9%

# getting classification Report
# convert test_prob into binary
test_data['test_pred_col'] = np.zeros(200)

test_data.loc[test_prob > optimal_threshold, 'test_pred_col']=1

classification_test = classification_report(test_data.clicked_on_ad, test_data.test_pred_col)
# test data giving excellent accuracy.

# training data accuracy check
train_prob = model.predict(train_data)

# getting FPR, TPR and threshold for train data
fpr2, tpr2, thresholds3 = roc_curve( train_data.clicked_on_ad, train_prob)

# ROC plot for train data
plt.plot(fpr2, tpr2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show() # covering most of the area.

train_auc = auc(fpr2, tpr2)
print(train_auc) # 99.02%

# creating new column to save predictions
train_data['train_pred_col'] = np.zeros(800)

# covert to binaray from train probabilities
train_data.loc[train_prob > optimal_threshold, 'train_pred_col'] = 1

# classification report for train data
classification_train = classification_report(train_data.clicked_on_ad, train_data.train_pred_col)

# get confusion matrix for both datasets
confusion_test = pd.crosstab(test_data.clicked_on_ad, test_data.test_pred_col )
confusion_test


confusion_train = pd.crosstab(train_data.train_pred_col, train_data.clicked_on_ad )
confusion_train

# accuracies
accuracy_test = (99+98)/200
accuracy_train = (396+379)/800

print('The training set accuracy is {}% and testing set accuracy is {}%'.format(round(accuracy_train*100,2), round(accuracy_test*100,2)))

# End of the script