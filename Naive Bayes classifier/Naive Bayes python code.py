#Naive Bayes
#1.) Prepare a classification model using Naive Bayes for Salary dataset, train and test datasets are given separately use both datasets for model building. 
#Answer;-
import numpy as np # linear algebra
import pandas as pd # data processing, 

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB

#Loading the data set
df = pd.read_csv(r'D:\360digi\assingment\Machine learning CT-Naive Bayes\SalaryData_Train.csv')
test = pd.read_csv(r'D:\360digi\assingment\Machine learning CT-Naive Bayes\SalaryData_Test.csv')
df1 = df.copy()
test1 = test.copy()
#checking for null values
df.isnull().sum()
test.isnull().sum()

x_train = df1.drop(['Salary'], axis = 1)
y_train = df1['Salary']

x_test = test1.drop(['Salary'], axis = 1)
y_test = test1['Salary']

# import category encoders
#pip install category_encoders
import category_encoders as ce
# encode remaining variables with one-hot encoding
x_train.columns
encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native'])

x_train = encoder.fit_transform(x_train)

x_test = encoder.fit_transform(x_test)

x_train.head()
x_train.shape
x_test.head()
x_test.shape

#scaling
cols = x_train.columns
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test)
x_train = pd.DataFrame(x_train, columns=[cols])
x_test = pd.DataFrame(x_test, columns=[cols])
x_train.head()
x_test.head()

# instantiate the model
gnb = GaussianNB()


# fit the model
gnb.fit(x_train, y_train)

#prediction for test data
y_pred = gnb.predict(x_test)

#Accuracy score for test data
accuracy_score(y_test, y_pred)

#prediction for train data
y_pred = gnb.predict(x_train)

#Accuracy score for train data
accuracy_score(y_train, y_pred)

###############################################################################
#Problem Statement: -
# This dataset contains information of users in social network. This social network has several business clients which can put their ads on social network and one of the Client has a car company who has just launched a luxury SUV for ridiculous price. Build the Bernoulli Naïve Bayes model using this dataset and classify which of the users of the social network are going to purchase this luxury SUV.
#Answer;-
df = pd.read_csv(r'D:\360digi\assingment\Machine learning CT-Naive Bayes\NB_Car_Ad.csv')
df.drop(['User ID'], axis = 1 ,inplace = True)
df.columns

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}).astype(int)

x = df.drop(['Purchased'], axis = 1)
y = df['Purchased']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

x_train.head()

cols = x_train.columns
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


x_train = pd.DataFrame(x_train, columns=[cols])
x_test = pd.DataFrame(x_test, columns=[cols])
x_train.head()
x_test.head()

# instantiate the model
gnb = GaussianNB()


# fit the model
gnb.fit(x_train, y_train)

#prediction for test data
y_pred = gnb.predict(x_test)

#Accuracy score for test data
accuracy_score(y_test, y_pred)

#prediction for train data
y_pred = gnb.predict(x_train)

#Accuracy score for train data
accuracy_score(y_train, y_pred)


###############################################################################
#Problem Statement: -
#In this case study you have been given with tweeter data collected from an anonymous twitter handle, with the help of Naïve Bayes algorithm predict a given tweet is Fake or Real about real disaster occurring. 
#Real tweet: - 1 and Fake tweet: - 0
#Answer;-

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set
df = pd.read_csv(r'D:\360digi\assingment\Machine learning CT-Naive Bayes\Disaster_tweets_NB.csv',encoding = "ISO-8859-1")

df.drop(['location','id'], axis = 1, inplace = True)
df.columns

df.fillna("Unknown", inplace = True)
df.isnull().sum()

# cleaning data 
import re # regular expression 
stop_words = []
# Load the custom built Stopwords

with open("D:/360 DATA TYPE/NLP- text mining/stopwords_en.txt","r") as sw:
    stop_words = sw.read()
  
stop_words = stop_words.split("\n")
   
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))


df.text = df.text.apply(cleaning_text)
type(df.text)
type(df)
# removing empty rows
df = df.loc[df.text != " ",:]

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts
# splitting data into train and test data sets 

df_train, df_test = train_test_split(df, test_size = 0.2)

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of email texts into word count matrix format - Bag of Words
df_bow = CountVectorizer(analyzer = split_into_words).fit(df.text)

# Defining BOW for all messages
df_matrix = df_bow.transform(df.text)

# For training messages
train_df_matrix = df_bow.transform(df_train.text)

# For testing messages
test_df_matrix = df_bow.transform(df_test.text)

# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(df_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_df_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_df_matrix)
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

df.columns
# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, df_train.target)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == df_test.target)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, df_test.target) 

pd.crosstab(test_pred_m, df_test.target)

# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == df_train.target)
accuracy_train_m

##################################################################################################################################33