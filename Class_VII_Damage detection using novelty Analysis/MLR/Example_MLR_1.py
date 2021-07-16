# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:12:23 2021

@author: Enrique GM
"""

import numpy as np
import matplotlib.pyplot as plt

npp = 3000
x = np.random.randn(npp,1)
y = x**2+np.random.randn(npp,1)

sort_index = np.argsort(x.T).T 
xsort = x[sort_index,0]
ysort = y[sort_index,0]


plt.figure(1)
plt.scatter(x,y, marker='o', s=2, c='b')

# First order
X = np.concatenate((np.ones((np.size(xsort,0),1)),xsort),axis=1)
XXT = np.matmul(X.T,X)
iXXT = np.linalg.pinv(XXT)
beta = np.matmul(np.matmul(iXXT,X.T),ysort)

# Second order
X2 = np.concatenate((X,xsort**2),axis=1)
XXT = np.matmul(X2.T,X2)
iXXT = np.linalg.pinv(XXT)
beta2 = np.matmul(np.matmul(iXXT,X2.T),ysort)

# Third order
X3 =  np.concatenate((X2,xsort**3),axis=1)
XXT = np.matmul(X3.T,X3)
iXXT = np.linalg.pinv(XXT)
beta3 = np.matmul(np.matmul(iXXT,X3.T),ysort)

# Fourth order
X4 =  np.concatenate((X3,xsort**4),axis=1)
XXT = np.matmul(X4.T,X4)
iXXT = np.linalg.pinv(XXT)
beta4 = np.matmul(np.matmul(iXXT,X4.T),ysort)

plt.figure(2)
plt.scatter(x,y, marker='o', s=2, c='b')
plt.plot(xsort,np.matmul(X,beta),'-b',linewidth=2)
plt.plot(xsort,np.matmul(X2,beta2),'-g',linewidth=2)
plt.plot(xsort,np.matmul(X3,beta3),'-r',linewidth=2)
plt.plot(xsort,np.matmul(X4,beta4),'--k',linewidth=2)
