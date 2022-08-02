# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 20:54:03 2022

@author: rohit
"""

# Logistic Regression Assignment on affairs data

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

data = pd.read_csv(r'D:\360digitmg\lectures\25 Logistic Regression\Assignment\Datasets_LR\Affairs.csv')
data1 = data.copy()

# as per quenstion lets convert naffairs into binary.
data.naffairs = np.where(data.naffairs != 0,1,0) # Replacing number other than zero to 1.
data.naffairs.value_counts() # 1: affairs, 0: No-affairs.

data.columns
data.drop('Unnamed: 0', axis=1, inplace=True)
data.shape  # (601, 18)

#EDA 
data.isna().sum()
data.duplicated().sum() # 378
data.drop_duplicates(keep='first', inplace=True)
data.shape # (223, 18)

for col in data.columns:
    print(col, data[col].nunique(), '\n', data[col].unique())
    print(" ")  # all columns have only two unique values.
data.var(axis=0)==0

# model Building 
import statsmodels.formula.api as smf 
f = '{}'.format('+'.join(data.columns))
ml1 = smf.logit('naffairs ~ kids+vryunhap+unhap+avgmarr+hapavg+vryhap+antirel+notrel+slghtrel+smerel+vryrel+yrsmarr1+yrsmarr2+yrsmarr3+yrsmarr4+yrsmarr5+yrsmarr6', data=data).fit()
# logit used for logistic regression.

# summary 
ml1.summary()
# we cannot have R^2 values here like we see that in Linear regression. 
# then how to know if model is good or not. Hence we will see summary2.
ml1.summary2() # AIC
# AIC = Akaike information criterion, lower the AIC value better the result.

# prediction
pred = ml1.predict(data.iloc[:, 1:])
# these are probability values, we have to cut them into binary.

from sklearn.metrics import roc_curve, auc 
fpr, tpr, thresholds = roc_curve(data.naffairs, pred) # y_true, y_score 
# roc only works with binary data.
# fpr = False positive rate, tpr= True positive rate
# threshold is nothing but cut-off value.
# for each threshold values we get different tpr and fpr values.
# we need more tpr and less fpr so we can get more area under curve (AUC).

# here we need high tpr and least fpr, hance we get more difference of tpr-fpr.
optimal_idx = np.argmax(tpr-fpr) 
# argmax() function returns indices of the max element of the array in a particular axis.
# this means it will go axis wise difference.
# Optimal_idx is basically row number at that differance. 

# Getting optimum value of threshold
optimal_threshold = thresholds[optimal_idx]
optimal_threshold # at 27th row we get 0.4295969944491264 value as our threshold or cut-off.
# it is a ideal cut-off we get.


# just getting number from zero to len(tpr).
i = np.arange(len(tpr)) 
roc = pd.DataFrame({'fpr':pd.Series(fpr, index=i),'tpr':pd.Series(tpr, index=i), '1-fpr':pd.Series(1-fpr, index=i), 
                    'tf':pd.Series(tpr - (1-fpr), index = i), 'threshold':pd.Series(thresholds, index=i)})

# Roc curve and AUC
# plot ROC
plt.plot(fpr, tpr); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate');plt.show()

 # area under the curve
roc_auc = auc(fpr, tpr)
print('Area under the ROC curve: %f' % roc_auc) # 0.652464 = 65% model performace.

# creating new columns and filling all the cells with zeros 
data['Pred']=np.zeros(223)

# taking threshold value and above the prob value will be treated as correct value
data.loc[pred > optimal_threshold, 'Pred'] = 1 # pred is for prob value and 2nd 'Pred' is column name.
# when prob value > optimal threshold we consider it as one otherwise zero, converting prob into binary.

# classification report (for evaluation) 
from sklearn.metrics import classification_report # to compare reprt of test and train data 
classification = classification_report(data['naffairs'], data['Pred']) # y_true, y_pred
classification
# here support is nothing but count

# Now we have optimum cut-off, and overall accuracy, etc... we will go for model building.

#### splitting the data
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size = 0.2) # 20 %test data


# model Building 
model = smf.logit('naffairs ~ kids+vryunhap+unhap+avgmarr+hapavg+vryhap+antirel+notrel+slghtrel+smerel+vryrel+yrsmarr1+yrsmarr2+yrsmarr3+yrsmarr4+yrsmarr5+yrsmarr6', data=train_data).fit()
model.summary() 
model.summary2()

# prediction 
test_pred = model.predict(test_data)


# creating new column for sorting predicted class of naffairs
# filling all the cells with zeros
test_data['test_pred']=np.zeros(45)

# taking threshold value as 'optimal_threshold' and above threshold probabilities value will be treated as 1.
test_data.loc[test_pred > optimal_threshold, 'test_pred']=1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['naffairs'])
confusion_matrix

accuracy_test = (14+11)/(14+11+15+5)
accuracy_test # 55.55% 

# classification Report
classification_test = classification_report(test_data['test_pred'], test_data['naffairs'])

# ROC and AUC
fpr, tpr, threshold = roc_curve(test_data['naffairs'], test_pred)

# olot ROC
plt.plot(fpr, tpr)
plt.xlabel('False + ve Rate')
plt.ylabel('True + ve Rate')
plt.show()

roc_auc_test = auc(fpr, tpr)
roc_auc_test # ~48% area under the curve.


# since we have got 55% accuracy on test set lets check for train set
# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1: 18])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(178)

# taking threshold value to convert into binary.
train_data.loc[train_pred > optimal_threshold, 'train_pred'] = 1

# confusion Matrix
confusion_train = pd.crosstab(train_data.train_pred, train_data.naffairs)
confusion_train

accuracy_train = (63+44)/(63+44+30+41)
accuracy_train # 60% 

print(accuracy_train*100)
print(accuracy_test*100)


# END of the script