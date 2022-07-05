# Singular-value decomposition
import numpy as np
from numpy import array
from scipy.linalg import svd # linalg = Linear algebra

# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])

A = array([[1,0,0,0,2], [0,0,3,0,0], [0,0,0,0,0], [0,4,0,0,0]])
print(A)

# SVD
U, d, Vt = svd(A)

print(U)
print(d)
print(np.diag(d))
print(Vt)


# Applying SVD on dataset
import os
os.chdir(r'D:\360digitmg\lectures\9 Dimension Reduction PCA')

# import data
import pandas as pd
data = pd.read_excel('University_Clustering/University_Clustering.xlsx')
data.head()

data = data.iloc[:,2:]
data.head()

from sklearn.decomposition import TruncatedSVD
# svd
svd = TruncatedSVD(n_components=3) #n_components means output columns
svd.fit(data)
result = pd.DataFrame(svd.transform(data))
result.head()

# don.t compare PCA results with SVD results. - Sharat sir
# we get different results in both cases.

result.columns = "pc0", "pc1", "pc2"

# Scatter diagram
import matplotlib.pylab as plt
plt.scatter(x = result.pc0, y = result.pc1)
