# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 21:05:22 2022

@author: rohit
"""

# Logistic Regression Assignment on bank data

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

data = pd.read_csv(r'D:\360digitmg\lectures\25 Logistic Regression\Assignment\Datasets_LR\bank_data.csv')
data1 = data.copy()
data.columns

# Rename columns to remove dots in the names
data.rename(columns={'joself.employed': 'joself_employed', 'joadmin.':'joadmin','joblue.collar':'joblue_collar'},inplace=True)

for col in data.columns:
    print(col, data[col].nunique(), '\n', data[col].unique())
    print(" ")  # all columns have only two unique values.
    
stats = data.describe()
data.default.value_counts()
data.housing.value_counts()
data.loan.value_counts()
data.previous.value_counts()

# remove defalut data
data.drop('default', axis=1, inplace=True)

#EDA 
data.isna().sum()
data.duplicated().sum()
data.drop_duplicates(keep='first', inplace=True)
data.shape # (45210, 31)

# scale the data
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
y = data.y
X = data.iloc[:,:30]
col_name = X.columns
data= pd.DataFrame(sc.fit_transform(X))
data.columns = col_name

# final data
data = pd.concat([data, y], axis=1)
data.y.value_counts()

# model Building 
import statsmodels.formula.api as smf 
f = '{}'.format('+'.join(data.columns))
ml1 = smf.logit('y ~ age+balance+housing+loan+duration+campaign+pdays+previous+poutfailure+poutother+poutsuccess+poutunknown+con_cellular+con_telephone+con_unknown+divorced+married+single+joadmin+joblue_collar+joentrepreneur+johousemaid+jomanagement+joretired+joself_employed+joservices+jostudent+jotechnician+jounemployed+jounknown', data=data).fit()

# logit used for logistic regression.
ml1.summary()
ml1.summary2() # AIC = 27787.5962

# prediction and getting probabilities
pred = ml1.predict(data.iloc[:, data.columns != 'y'])

# getting FPR, TPR and threshold
from sklearn.metrics import roc_curve, auc, classification_report
fpr, tpr, thresholds = roc_curve(data.y, pred)

# throwing error since there are is NAN values.
pred.isna().sum()
data.y.isna().sum()
print(pred[45210]) 
pred.fillna(pred.mean(), inplace=True)
data.y.fillna(data.y.mode()[0], inplace=True)
# again run fpr, tpr thresholds command line

# finding optimal threshold
optimal_idx = np.argmax(tpr-fpr) # getting index of max difference

optimal_threshold = thresholds[optimal_idx]
optimal_threshold # 0.11003301185728781

# plot roc curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show() # covering most of the area.

# getting numrical value of AUC
roc_auc = auc(fpr, tpr)
roc_auc # 0.7908784485579741

# Create column of all zeros
data['pred_col'] = np.zeros(45211)

# coverting prob in pred into 0 and 1 using optimal_threshold
data.loc[pred>optimal_threshold, 'pred_col']=1
data.pred_col.value_counts()

# getting classification report
classificaiton = classification_report(data.y, data.pred_col)

# every thing is fine. we have optimal threshold.
#### splitting the data
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size = 0.2) # 20 %test data

# model Building 
model = smf.logit('y ~ age+balance+housing+loan+duration+campaign+pdays+previous+poutfailure+poutother+poutsuccess+poutunknown+con_cellular+con_telephone+con_unknown+divorced+married+single+joadmin+joblue_collar+joentrepreneur+johousemaid+jomanagement+joretired+joself_employed+joservices+jostudent+jotechnician+jounemployed+jounknown', data=train_data).fit()
model.summary() 
model.summary2() # AIC = 22403.1246

# getting test data probabilities
test_prob = model.predict(test_data)

# getting FPR, TPR and threshold for test data
fpr1, tpr1, thresholds1 = roc_curve( test_data.y, test_prob)

# ROC plot for test data
plt.plot(fpr1, tpr1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show() # covering most of the area.

test_auc = auc(fpr1, tpr1)
print(test_auc) # 7917197594616949

# getting classification Report
# convert test_prob into binary
test_data['test_pred_col'] = np.zeros(9043)

test_data.loc[test_prob > optimal_threshold, 'test_pred_col']=1

classification_test = classification_report(test_data.y, test_data.test_pred_col)
# test data giving excellent accuracy.

# training data accuracy check
train_prob = model.predict(train_data)

# getting FPR, TPR and threshold for train data
fpr2, tpr2, thresholds3 = roc_curve( train_data.y, train_prob)
# check null
data.y.isna().sum()
train_prob.isna().sum()
train_prob.fillna(train_prob.mean(), inplace=True)



# ROC plot for train data
plt.plot(fpr2, tpr2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show() # covering most of the area.

train_auc = auc(fpr2, tpr2)
print(train_auc) # 7905841297188351

# creating new column to save predictions
train_data['train_pred_col'] = np.zeros(36168)

# covert to binaray from train probabilities
train_data.loc[train_prob > optimal_threshold, 'train_pred_col'] = 1

# classification report for train data
classification_train = classification_report(train_data.y, train_data.train_pred_col)

# get confusion matrix for both datasets
confusion_test = pd.crosstab(test_data.y, test_data.test_pred_col )
confusion_test


confusion_train = pd.crosstab(train_data.train_pred_col, train_data.y )
confusion_train

# End of the script