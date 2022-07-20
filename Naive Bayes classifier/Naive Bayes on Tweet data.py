# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 19:14:42 2022

@author: rohit
"""

# Assignment on Naive Bayes theorem on tweet data.

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 

# load the data
tweet = pd.read_csv(r'D:\360digitmg\lectures\16 ML Classifier Technique-Naive Bayes\Assignment\dataset\Disaster_tweets_NB.csv',encoding = "ISO-8859-1")

# make a copy of data
tweet2=tweet.copy()

# here label is Target column. and text is input column.

# Cleaning the data
import re

def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower() 
    # ^(called as negation), this symbol shows,except this (i.e. A-Za-z) convert  all to empty spaces. then convert all to lower case.
    i = re.sub("[0-9" "]+"," ",i) 
    # no cap symbol means we change all numbers to spaces. if want work only with numbers then put cap symbol. it will keep only numbers rest will converted to spaces.  
    # we are not taking this beacause, all number are changed to spaces by first line.
    w = []
    for word in i.split(" "):
        if len(word)>3: # taking words whose length is more than 3. ( words like am, is, are, i, an, etc are removed)
            w.append(word)
    return (" ".join(w))


tweet.text = tweet.text.apply(cleaning_text)

# removing empty rows
tweet = tweet.loc[tweet.text != " ", :]

# CountVectorizer 
# convert a collection of text documents to a matrix to token counts
# split data into train test set.
from sklearn.model_selection import train_test_split

tweet_train, tweet_test = train_test_split(tweet, test_size=0.2)


# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of tweet text into word count matrix format - Bag of Words
tweet_bow = CountVectorizer(analyzer=split_into_words).fit(tweet.text)

# definig BOW for all tweets
all_tweet_matrix = tweet_bow.transform(tweet.text)


# For training tweet
train_tweet_matrix = tweet_bow.transform(tweet_train.text)

# For testing tweet
test_tweet_matrix = tweet_bow.transform(tweet_test.text)


# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_tweet_matrix)
# here we create DTM, each message willbe  at document and tokens/keywords be at terms.


# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_tweet_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_tweet_matrix)  # here we get normalized values.
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB
# this multiNB ML is used for didcrete features

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, tweet_train.target)
# here all calculations of spam or ham mail is done using conditional probability.

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == tweet_test.target)
# since the label is discrete data, we just compare prediction to actual values.
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, tweet_test.target) # another way to get accuracy.
# both accuracy calculating techniques do same process of comparison.

# method 1 of cross validation for test accuracy
# Training Data accuracy. it is just for cross validation (check accuracy of train data).
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == tweet_train.target)
accuracy_train_m

# both test and train data almost have same accuracy. 

# method 2 for cross validation
# kind of confusion matrix
pd.crosstab(test_pred_m, tweet_test.target)

# type 1 error: spam predicted as ham.
# type 2 erroe: ham predicted as spam. this is more severe than type 1.
# there is probability that this ML model may put a very important mail into spam.
# having accuracy more 90% does not mean our model is good and it the best to deploy.
# we have to check for type 1 & 2 errors also. and how to can we make these errors as low as possible.
# we always have reduce type 2 error, since cost associated with this error is more  than typ1 error.
# try to make type2 error to zero.

# cross table for train data
pd.crosstab(train_pred_m, tweet_train.target)
# here type 2 error is zero, this one is good. 



# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.

# formula:
    # P(w|spam) = (No.of spam with w + alpha) / (total no. of spam + K(alpha))
    # K = total number of words in mail that need to be classified.
    # w = word 
# alpha can be any number. eg. 1 or 100 or 1000 etc.
# probability of mail being spam given that new word is present is zero. 
# hence to avoid this zero value. we can change alpha value. 
# aloha vlaue is not fixed, we have to try for optimum values of alpha.
# set one alpha value then check accuracy and cross table. if these two values to makes feel that model is good then keep that alpha value.
# default aplha is 1.
classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(train_tfidf, tweet_train.target)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == tweet_test.target)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, tweet_test.target) 

pd.crosstab(test_pred_lap, tweet_test.target)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == tweet_train.target)
accuracy_train_lap


############ End of the script ######### 























