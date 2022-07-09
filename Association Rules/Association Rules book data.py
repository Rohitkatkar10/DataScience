# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 13:04:03 2022

@author: rohit
"""

# Assignment on Association Rule: Book data 

# importing necessary libraries.

import os
os.chdir(r'D:\360digitmg\lectures\11 DM- Unsupervised Learning Association rules\Assignment\Datasets_Association Rules')

import pandas as pd
book = pd.read_csv('book.csv')

book.info()
stats = book.describe() # maximum values are 1 and min are 0. for all variables.

# check missing 
book.isna().sum()  # No Missing

# check duplicates
book.duplicated().sum()  # since data is in binary there can be duplicates. hence not removing them.

# plot the graph for visualization 
book.sum()
import matplotlib.pyplot as plt

plt.bar(x=list(range(0,11)),height=book.sum(), color= ['b','r','y','g','m','c','b'])
plt.xticks(list(range(0,11), ), book.columns)
plt.xlabel("itmes")
plt.ylabel('Count')
plt.show()

# from graph: cook books and child books are most sold books.

# apply association rules 
from mlxtend.frequent_patterns import apriori, association_rules


# set Minimum support criteria using apriori function 
book_apriori = apriori(book, min_support=0.03,max_len=4, use_colnames=True)
# If min_support = 0.3 then we get only rows. hence need to reduce the values to 0.03, so we can get many rows.
# max_len is 'k' value that is how many item we want to take into consideration for min_support.
# if max_len = 4, then it will take item once and see if it satisfy minumum support if yes then it will keep that item into list. 
# if no then those item are not taken into list. then all these passed item are taken into pairs then see min_support if they passes,'
# then take these to passed pairs and make it trio and see if then pass the min_support criteria, in this way we go upto k values pairs.
# in this case max 4 items will have to pass the minimum support critetia. 

# get descending order on the basis of support.
book_apriori.sort_values('support', ascending=False, inplace=True)

# Association Rules
rules = association_rules(book_apriori, metric='lift', min_threshold=1)
rules.head()
# if lift > min_threshold, then it is good rule, else bad rule. 

# Now see first two rows of rules DataFrame, it is just interchange of antecedent and consequent and their lift value is same.
# this is called profusion.

# to address profusion problem.
def to_list(i):
    return(sorted(list(i))) 

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)
# this above line will make list of items in antecedents and consequents, will sort it. 
# and add these two to make new series in new variable.

# now we will sort these items in series.
ma_X = ma_X.apply(sorted) # now see this repetation when sorted alphabatically.

# make list of ma_X
rules_sets = list(ma_X)

# Remove duplicates from rule_sets.
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
# duplicates eliminated using set. 

# To find the index of duplicate items so we can delete duplicates in support, lift, etc columns also.

index_rules=[]

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i)) # appending index number of that list in unique_rules_sets.

# Getting rules without any redundancy
rules_no_redundancy = rules.iloc[index_rules, :]

final_rules = rules_no_redundancy.sort_values('lift', ascending =False)

# see first 20 observation
first_ten = final_rules.head(20)
last_ten = final_rules.tail(10)

#####################################################################
############ THE END OF SCRIPT ##############



