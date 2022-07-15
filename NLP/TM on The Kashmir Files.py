# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:18:37 2022

@author: rohit
"""



# to install the imdbpy library, just including it for noob-friendliness     
# pip install imdbpy

# https://stackoverflow.com/questions/59969327/is-there-a-way-to-extract-imdb-reviews-using-imdbpy

# importing imdb library to process the data
from imdb import IMDb 
# Creating instance 
ia = IMDb()

# Now we will take review on a movie:The Kashmir Files
# link: https://www.imdb.com/title/tt10811166/reviews?ref_=tt_urv
# To get the data, we need imdb code of the movies. it is in the link after 'tt'
# movies code = 10811166

theMatrix = ia.get_movie('10811166') # getting information from the page.

theMatrix.current_info # we will get these information. see console.

theMatrix = ia.get_movie('10811166',['reviews'])  # we will get review from imdb page. saving to variable.

theMatrix.current_info # returns what we got in that variable.

# see the output of the following command. we are getting list of dictionaries.
# dictionaries having keys like content i.e. review, rating, helpful, auther, date of review, etc.
theMatrix['reviews'] 

# we need 0th item from list
theMatrix['reviews'][0] # gives first dictionary. 
# understanding clearly. what is in the dictionary.

type(theMatrix['reviews'][0]) # from here we can say this surely is dict.

# saving to variable
rev_dic = theMatrix['reviews'][0]

# getting review contenct from first dictionary 
print(rev_dic.get('content'))

# Now we will want only reviews.
# saving to variable
list_rev = [theMatrix['reviews']] # this list of dict.


for i in theMatrix['reviews']:
    print(i.get('content'))
    break

# save all reviews in one list.
files_reviews = []

for i in theMatrix['reviews']:
    ip=[]
    ip.append(i.get('content'))
    files_reviews = files_reviews+ip

# writing reviews in text file
with open('files_reviews', 'w', encoding='utf8') as output:
    output.write(str(files_reviews))
    
    
# joining all the reviews into into single paragraph
ip_rev_string = " ".join(files_reviews)

import nltk
import re
# from nltk.corpus import stopwrods

# Removing unwanted symbols incase if exists
ip_rev_string = re.sub("[^A-Za-z" "]+"," ", ip_rev_string).lower()
# ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string)

# words the contained in shrimad Bhagvat gita reviews
ip_reviews_words = ip_rev_string.split(" ") # there is a empty character at first


# Tfidf
from sklearn.feature_extraction.text import TfidfVectorizer 
# vectorizer = TfidfVectorizer( ip_reviews_words, use_idf=True, ngram_range=(1,3)) # Not working
vectorizer = TfidfVectorizer( use_idf=True, ngram_range=(1,3))
X = vectorizer.fit_transform(ip_reviews_words)

with open(r'D:\360digitmg\lectures\14 Text Mining Sentiment Analysis\Datasets NLP\stop.txt','r') as sw:
    stop_words = sw.read()

stop_words = stop_words.split('\n')

# remove stop words
ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]

# joining all reviews into a paragraph
in_rev_string = " ".join(ip_reviews_words)  
 
# wordcloud can be performed on the string input
# carpus level word cloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud_ip = WordCloud(background_color='White',
                         width=1800,
                         height=1400).generate(ip_rev_string)   
     
plt.imshow(wordcloud_ip) # cloud with unigrams.



# positive words # Choose the path for +ve words stored in system
with open(r"D:\360digitmg\lectures\14 Text Mining Sentiment Analysis\Datasets NLP\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")

# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)
plt.figure(2)
plt.imshow(wordcloud_pos_in_pos)

# negative words Choose path for -ve words stored in system
with open(r"D:\360digitmg\lectures\14 Text Mining Sentiment Analysis\Datasets NLP\negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n")

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_neg_in_neg)

########### End of the script ########
