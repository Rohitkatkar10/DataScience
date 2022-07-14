# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:33:56 2022

@author: rohit
"""

# Network Analytics on Flight hault data.
import pandas as pd
import networkx as nx

halt = pd.read_csv(r'D:\360digitmg\lectures\13 Network Analytics\Assignment\Datasets_Network Analytics\flight_hault.csv')
halt.columns
# see clearly there are no column names. thsese are the data points.
# Add columns names from assignment  document

halt.columns = ["ID","Name","City","Country","IATA_FAA","ICAO","Latitude","Longitude","Altitude","Time","DST","Tz database time"]

# now add the initial columns to dataframe
halt.loc[0] = ['1', 'Goroka', 'Goroka.1', 'Papua New Guinea', 'GKA', 'AYGA',
       '-6.081689', '145.391881', '5282', '10', 'U', 'Pacific/Port_Moresby']

halt.info() # nan values in IATA FAA column

halt.IATA_FAA.fillna(halt.IATA_FAA.mode()[0], inplace=True) # since there are two modes, select zeroth mode.

# graph 
g = nx.Graph() # empty graph
g = nx.from_pandas_edgelist(halt, source='IATA_FAA', target='ICAO' )
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