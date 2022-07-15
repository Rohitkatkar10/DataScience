# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 18:15:42 2022

@author: rohit
"""

# Amazon Web extraction for Shrimad Bhagvat Gita.

# import libraries
import requests # to extract content from url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content

import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# creating empty list for reviews
gita_reviews = []

for i in range(1,21):
    ip=[]
    url="https://www.amazon.in/Bhagvad-gita-as-english-new/product-reviews/9384564192/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
    response = requests.get(url)
    soup = bs(response.content, "html.parser") # creating soup object to iterate over the extracted content 
    reviews = soup.find_all("span",attrs={"class","a-size-base review-text review-text-content"}) # Extracting the content under specific tags  
    for i in range(len(reviews)):
        ip.append(reviews[i].text)
    gita_reviews = gita_reviews+ip # adding the reviews of one page to empty list which in future contains all the reviews
    
# writing reviews in text file
with open('gita_reviews', 'w', encoding='utf8') as output:
    output.write(str(gita_reviews))
    
# joining all the reviews into into single paragraph
ip_rev_string = " ".join(gita_reviews)

import nltk
# from nltk.corpus import stopwrods

# Removing unwanted symbols incase if exists
ip_rev_string = re.sub("[^A-Za-z" "]+"," ", ip_rev_string).lower()
# ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string)

# words the contained in shrimad Bhagvat gita reviews
ip_reviews_words = ip_rev_string.split(" ") # there is a empty character at first
ip_reviews_words = ip_reviews_words[1:] 


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

# wordcloud with bigrams
nltk.download('punkt')
from wordcloud import WordCloud, STOPWORDS    

WNL = nltk.WordNetLemmatizer() 

# lowercase and tokenize
text = ip_rev_string.lower()    

# Remove single quote early since it causes problems with tokenizer
text = text.replace("'","")

tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)

# Remove extra chars and remove stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]



# Create a set of stopwords
stopwords_wc = set(STOPWORDS)
# customised_words = ['price', 'great'] # If you want to remove any particular word form text which does not contribute much in meaning
# new_stopwords = stopwords_wc.union(customised_words)

# Remove stop words
text_content = [word for word in text_content if word not in stopwords_wc]

# take only non-empty entries 
text_content = [s for s in text_content if len(s) != 0]

# Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]

nltk_tokens = nltk.word_tokenize(text)
bigram_list = list(nltk.bigrams(text_content))
print(bigram_list)

dictionary2 = [' '.join(tup) for tup in bigram_list] # list of all
print(dictionary2)

# using count vectorizer to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(2,2)) # only bigrams
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])


# Generating wordcloud
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 200
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=stopwords_wc)
wordCloud.generate_from_frequencies(words_dict)

plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()


########### End of the script ########

