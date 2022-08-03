# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:28:40 2022

@author: rohit
"""


### Multinomial Regression ####
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mode = pd.read_csv(r"D:\360digitmg\lectures\26 Multinomial Regression\Assignment\Datasets_Multinomial\loan.csv")
mode1 = mode.copy()
mode.info()
null = mode.isna().sum() # there are many empty columns. 
all_col = mode.columns

# get all empty and null values > 10,000 columns
fake_col = list()
for index in null.index:
    if null[index] > 10000:
        fake_col.append(index)

# Now Remve these empty columns.
mode.drop(fake_col,  axis=1, inplace=True)
# now check null values
mode.isna().sum()

# fill null values
for col in mode.columns:
    if mode[col].isna().sum() != 0:
        mode[col].fillna(mode[col].mode()[0], inplace=True)
      
stats = mode.describe()
mode.prog.value_counts() # target variable

# remove unnecessary columns
col = ['member_id','id']
col
mode.drop(col, axis=1, inplace=True)
mode.columns

cat_col = [col for col in mode.columns if mode[col].dtype == 'O']
num_col = [col for col in mode.columns if mode[col].dtype != 'O' ]
for col in cat_col:
    print(col, mode[col].nunique(),mode[col].dtype,'\n',mode[col].unique())
    print(" ")

cat_col_keep = ['term','grade','emp_length','home_ownership','verification_status','loan_status']
cat_col_remove = [col for col in cat_col if col not in cat_col_keep]
mode.drop(cat_col_remove, axis=1, inplace=True)

# check variance
mode.var(axis=0) == 0
mode.annual_inc.var() == 0

# remove zero variance col.
for col in num_col:
    if mode[col].var() == 0:
        mode.drop(col, axis=1, inplace=True)


mode.info()

# load_status is output column.
mode.loan_status.value_counts()

# check duplicates
mode.duplicated().sum() # NO

# label encoding
from sklearn.preprocessing import LabelEncoder 
# creating instance for label encoder
labelencoder = LabelEncoder()
X = mode.iloc[:, mode.columns != 'loan_status']
y = mode.loan_status

for col in X.columns:
    if X[col].dtype == 'O':
        X[col] = labelencoder.fit_transform(X[col])
        
# mode = pd.concat([X, y], axis=1)
# scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_col = X.columns
X_ss = pd.DataFrame(scaler.fit_transform(X))  # column names are numbers.
X_ss.columns = X_col # for Hierarchical


# apply PCA to reduce stress on system 

from sklearn.decomposition import PCA
pca = PCA(n_components=28) # compenents should be equal to number of original columns.
pca_value = pca.fit_transform(X_ss)

pca.components_
pca.components_[0] # First row.

# Amount of variance that each variance explain is
var = pca.explained_variance_ratio_
var

# take cumulative sum of variance
var1 = np.cumsum(np.round(var, decimals=4)*100) #taking % of variance (in cumulative)
var1  #  decimal=4 means e.g. 24.13 ( total digit are 4)


# variance plot for PCA component obtained.
# graph is opposite to scree plot or elbo curve (used in K means clustering)
plt.plot(var1, color='red')
plt.xlabel("Number of PCA's")
plt.ylabel("Information in percentage")
plt.title('Variance Plot')
plt.show()

# PCA score 
pca_value # convert var1 from array to dataframe

pca_data = pd.DataFrame(pca_value) # No column names

# Give names to columns
pca_data.columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20','PC21','PC22','PC23','PC24','PC25','PC26', 'PC27','PC28']

# from plot I decide to take only forst 17 columns whos give 97% information.
mode = pd.DataFrame(pca_data.iloc[:,:17]) # considering only first three columns as mentioned in question.

# add output col in this.
mode = pd.concat([mode, y], axis=1)


# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(mode) # Normal
sns.pairplot(mode, hue = "load_status") # With showing the category of each car choice in the scatter plot

# Correlation values between each independent features
mode.corr()
mode.info()

# declare features and label
X = mode.iloc[:, mode.columns != 'loan_status']
# label already declared


# data split
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(X_train, y_train)
# help(LogisticRegression)

test_predict = model.predict(X_test) # Test predictions

# Test accuracy 
accuracy_score(y_test, test_predict)

train_predict = model.predict(X_train) # Train predictions 
# Train accuracy 
accuracy_score(y_train, train_predict) 

# both accuracies are almost same.

# End of the script
