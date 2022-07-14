# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:19:49 2022

@author: rohit
"""

# Network Analytics on Facebook data.

# refer below link to understand the concept.
# https://www.geeksforgeeks.org/introduction-to-social-networks-using-networkx-in-python/



import pandas as pd
import networkx as nx
import numpy as np

facebook = pd.read_csv(r'D:\360digitmg\lectures\13 Network Analytics\Assignment\Datasets_Network Analytics\facebook.csv')
facebook.columns = [0,1,2,3,4,5,6,7,8]

# graph
g = nx.Graph() # creating empty graph. 
g=nx.from_pandas_adjacency(facebook)


pos = nx.circular_layout(g)
nx.draw_networkx(g, pos, node_size=25, node_color='blue') 
# node_size is just diameter of the node circle.

