# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:23:20 2022

@author: rohit
"""
import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 

import matplotlib.pyplot as plt
from wordcloud import WordCloud

url = "https://en.wikipedia.org/wiki/Rama"

response = requests.get(url)
soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 


print(soup.get_text())
reviews = soup.get_text()
    

# getting all text from page saving this to text list.
text_main = rev.split('\n')
text=[]
for i in text_main:
    if len(i)==0:
        continue
    else:
        text.append(i)
        
with open("LordRam.txt","w",encoding='utf8') as output:
    output.write(str(text))

# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(text)



import nltk
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
        







