# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:34:25 2022

@author: rohit
"""

# Assignment on Association Rule: Movies data 

# importing necessary libraries.

import os
os.chdir(r'D:\360digitmg\lectures\11 DM- Unsupervised Learning Association rules\Assignment\Datasets_Association Rules')

import pandas as pd
movies = pd.read_csv('my_movies.csv')
# from dataframe, it is clear that, first five columns have data rest columns have dummy columns for movies data.

movies = movies.iloc[:,:5] # column only have mmovie name, binary data columns removed.
movies = movies.fillna(value=' ') # replace nan with space.

# we need to count frequency of movies in dataframe. lets convert all rows into lists 
mov_array = movies.to_numpy() # converted from dataframe to array
mov_list = mov_array.tolist()  # converte from array to list

# mov_list = movies.to_numpy().tolist()   # this also works.

# lets make one list for all movie names.
movie_names = [ name for movnames in mov_list for name in movnames] # space is there.

# remove "nan" word from movie_names
movie_name = [i for i in movie_names if i != ' '] # Remove space from list


# frequency of movie names
from collections import Counter

mov_frequencies = Counter(movie_name) # it is kind of dictionary
# this data is to small to plot, we are getting frequency of each item here in easy.
# Hence no plot.

# convert mov_name list to dataframe
mov_df = pd.DataFrame(pd.Series(movie_name))
# No column name is there.
mov_df.columns = ['movie_names']

# Creating a dummy columns for each item in transactions.. using col Name as item names.
mov_df = pd.get_dummies(mov_df)  # got dummy columns

# Apriori.
from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets = apriori(mov_df, min_support=0.0075, max_len=10, use_colnames=True) #min support is 75 times. (75/10000) by supprt formula.

# Most Frequent item sets based on support.
frequent_itemsets.sort_values('support', ascending = False, inplace=True)
# sorting on the basis of 'support' columns.


# Association Rules.
rules = association_rules(frequent_itemsets, metric='lift',min_threshold=0.0001) 
rules.head(20)
rules.sort_values('lift', ascending = False).head(20)

# Getting no rules. 