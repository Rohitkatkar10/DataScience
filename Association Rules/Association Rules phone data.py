# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 21:21:27 2022

@author: rohit
"""

import os
os.chdir(r'D:\360digitmg\lectures\11 DM- Unsupervised Learning Association rules\Assignment\Datasets_Association Rules')

import pandas as pd
phone = pd.read_csv('myphonedata.csv') # only 11x9 

# from dataframe, it is clear that, first five columns have data rest columns have dummy columns for phone data.
phone = phone.iloc[:,:3] # column only have mmovie name, binary data columns removed.



# since the data is in dataframe. 
phone_df = pd.get_dummies(phone)


# Apriori.
from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets = apriori(phone_df, min_support=0.075, max_len=4, use_colnames=True) #min support is 75 times. (75/10000) by supprt formula.

# Most Frequent item sets based on support.
frequent_itemsets.sort_values('support', ascending = False, inplace=True)
# sorting on the basis of 'support' columns.


# Association Rules.
rules = association_rules(frequent_itemsets, metric='lift',min_threshold=1) 
rules.head(20)
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

