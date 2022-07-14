# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:21:40 2022

@author: rohit
"""


# Network Analytics on Flight hault data.
import pandas as pd
import networkx as nx
import numpy as np

connect = pd.read_csv(r'D:\360digitmg\lectures\13 Network Analytics\Assignment\Datasets_Network Analytics\connecting_routes.csv')
connect.columns

connect['Unnamed: 6'].value_counts() # there are values other than nan.
connect['0'].value_counts() # there are values other than nan.
connect.drop(['0'], axis=1, inplace=True) # Not useful column


# give names to these columns from assignment document.
connect.columns = ['Flight', 'ID', 'MainAirport','Main Airport ID','Destination','Destination ID', 'Haults', 'Machinary']

# add the row that was column name earlier.

connect.loc[0] = ['2B', '410', 'AER', '2965', 'KZN', '2990',np.nan , 'CR2']


# graph 
g = nx.Graph() # empty graph
g = nx.from_pandas_edgelist(connect, source='MainAirport', target='Destination' )
print(nx.info(g))


# degree centrality 
d = nx.degree_centrality(g)
print(d)

pos = nx.spring_layout(g)
nx.draw_networkx(g, pos, node_size=25, node_color='blue')  # these two commands take time to execute.


# closeness centrality
closenes = nx.closeness_centrality(g)
print(closenes)


# Betweeness Centrality 
b = nx.betweenness_centrality(g) # Betweeness_Centrality
print(b)


########## End of the Script ####

