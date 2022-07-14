# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:33:11 2022

@author: rohit
"""


import pandas as pd
import networkx as nx
import numpy as np

linkedin = pd.read_csv(r'D:\360digitmg\lectures\13 Network Analytics\Assignment\Datasets_Network Analytics\linkedin.csv')

# skip this line to see what happen for first time.
linkedin.columns = [0,1,2,3,4,5,6,7,8,9,10,11,12]

# graph
g = nx.Graph() # creating empty graph. 
g=nx.from_pandas_adjacency(linkedin)


pos = nx.circular_layout(g)
nx.draw_networkx(g, pos, node_size=25, node_color='blue') 
# node_size is just diameter of the node circle.


################## End of the script   ###############