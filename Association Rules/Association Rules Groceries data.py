# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:53:28 2022

@author: rohit
"""

# Assignment on Association Rule: Groceries data 

# importing necessary libraries.

import os
os.chdir(r'D:\360digitmg\lectures\11 DM- Unsupervised Learning Association rules\Assignment\Datasets_Association Rules')

import pandas as pd
groceries = pd.read_csv('groceries.csv') # Not possible to open the file using pandas.

# use open command to open the csv file.
with open('groceries.csv') as f:
    groceries = f.read()   

# need to preprocess the data, since these data is not in proper csv format. 
# it is shows a string with more than 50,000 character in it. Each line is shows items purchased by every customer.
# first will separate items purchased by each customer, then how many items are there to sell and how many times each item is sold.

# split one large string at newline into small lists.
groceries = groceries.split('\n')

# now make a seperate list for each customer
customer_list = []  # it will be list of lists.
for i in groceries:
    customer_list.append(i.split(',')) # split will convert item into list, we will get 9836 lists.
    
# frequency of items purchased.
all_groceries_list = [i for item in customer_list for i in item] # there are more than 43K item in list.

# count item purchases
from collections import Counter
item_frequency = Counter(all_groceries_list) # there is space and its count is one. Need to remove that empty space.

print(type(item_frequency)) # it has key-value pair like dictionary. this key-value pair is also called as 'items'. 

# sort item ( defalut= ascending order)
item_frequencies = sorted(item_frequency.items(), key = lambda x: x[1])
# here 'key' argument: sorting is done on the basis of the key.
# lambda x:x[1] means it is a function and x is a item and x[0]=key and x[1]=value of items.
# this means sorting of ITEMS is done on the basis on values. 

# sort purchsed item(key only) and frequencies(value only) in descending order seperatly.
items = list(reversed([i[0] for i in item_frequencies])) # here 'whole milk' is most sold item, will come first.
frequencies = list(reversed([i[1] for i in item_frequencies])) # 

# visualize 10 the most sold items.
import matplotlib.pyplot as plt

plt.bar(x=list(range(0,11)),height=frequencies[:11] , color=['r','b','y','g','k','m','c'])
plt.xticks(range(0,11), items[:11])
plt.xlabel('Most sold Items')
plt.ylabel('Frequency')
plt.title('Top 10 most sold items')
plt.show()

# Clearly, milk is most sold, followed by vegetable, rolls, soda, so on.

# Now Create DataFrame for the transaction of the data.
groceries_series = pd.DataFrame(pd.Series(customer_list))

# Name the column
groceries_series.columns = ['Transactions'] # there is space in transaction  columns at end.
groceries_series = groceries_series.iloc[:9835, :]

# creating dummy variabels for each item in data. using item as a columns name.
groceries_data = groceries_series.Transactions.str.join(sep='*').str.get_dummies(sep='*')
groceries_data.columns

#  Apply Apriori and Association rules.
from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets = apriori(groceries_data, min_support=0.01, max_len=4, use_colnames=True)
# we will keep minimum support = 1 out of 100 = 0.01

frequent_itemsets.sort_values('support', ascending=False, inplace=True) # sorting on the basis of support. 

# Association rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

# Now see first two rows of rules DataFrame, it is just interchange of antescedent and conscequent and their lift value is same.
# To address profusion (duplication of entry in X)
def to_list(i):
    return(sorted(list(i)))
    
x = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)
x = x.apply(sorted)

# make list 
rule_sets = list(x)

unique_rule_sets = [list(m) for m in set(tuple(i) for i in rule_sets)] # set avoids duplicate values.

# To find the index of duplicate items so we can delete duplicates in support, lift, etc columns also.
index_rule = []

for i in unique_rule_sets:
    index_rule.append(rule_sets.index(i))

# Getting rules without any redundancy
rules_no_redundancy = rules.iloc[index_rule, :]

# Sorting them with respect to list and getting top 10 rules 
r = rules_no_redundancy.sort_values('lift', ascending = False)
top_ten = r.head(10)

#####################################################################
############ THE END OF SCRIPT ##############



