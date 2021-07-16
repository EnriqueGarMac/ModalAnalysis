# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:49:05 2021

@author: Enrique GM
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy.matlib
from sklearn.decomposition import PCA
from numpy.linalg import eig


# PCA example


dt = 1/800;      # Time step
T = 20;          # Time length
N = int(T/dt)
t = np.linspace(0,N*dt,N, endpoint=True)

# Generation of variables
signal = np.sin(t*2*np.pi*10);         
noise = np.random.multivariate_normal(np.array([0,0]),np.array([[1,0.01],[0.01,0.05]]),size=N);
var1 = signal+noise[:,0];          # Variable 1
var2 = 10+var1+noise[:,1];         # Variable 2

plt.figure(1)
plt.plot(t,var1,label='Var 1')
plt.plot(t,var2,label='Var 2')
plt.xlabel('Time [s]')

plt.figure(2)
ax = plt.gca()
plt.plot(var1,var2,'x')
plt.xlabel('Var 1')
plt.ylabel('Var 2')
ax.set_aspect('equal', 'box')

# PCA analysis

# Normalization
x = np.array([var1,var2]).T
meanval = np.mean(x,axis=0);
stadval = np.std(x,axis=0);
x = (x-np.matlib.repmat(meanval,np.size(x,0),1))/np.matlib.repmat(stadval,np.size(x,0),1);

plt.figure(3)
plt.plot(t,x[:,0],label='Var 1')
plt.plot(t,x[:,1],label='Var 2')
plt.xlabel('Time [s]')


plt.figure(4)
plt.plot(x[:,0],x[:,1],'x')
plt.xlabel('Var 1')
plt.ylabel('Var 2')
ax.set_aspect('equal', 'box')

pca = PCA(2)
pca.fit(x)
print(pca.components_)
print(pca.explained_variance_)

COV = np.matmul(x.T,x)
print(COV/(N-1))
COV = np.cov(x.T)
print(COV)
D, V = eig(COV)

sort_index = np.flip(np.argsort(D))

latent2 = D[sort_index];
coeff2 = V[:,sort_index];
D = latent2;
explainedvar = 100*D/np.sum(D);
score2 = np.matmul(x,coeff2)

plt.figure(5)
ax = plt.gca()
plt.plot(x[:,0],x[:,1],'x')
plt.plot([0,V[0,0]*2*D[0]],[0,V[1,0]*2*D[0]],'r',LineWidth=2)
plt.plot([0,V[0,1]*20*D[1]],[0,V[1,1]*20*D[1]],'r',LineWidth=2)
ax.set_aspect('equal', 'box')
plt.xlabel('Var 1')
plt.ylabel('Var 2')


plt.figure(6);
plt.bar([1,2],explainedvar)
plt.xticks([1,2], ('PC1', 'PC2'))
plt.ylabel('Explained variance [%]')


# Dimension reduction
ll = 1
Z_vect=np.matmul(x,coeff2)
score2=Z_vect[:,np.arange(0,ll,1)];
reconstr=np.matmul(score2,coeff2[:,np.arange(0,ll,1)].T)
reconstr = reconstr*np.matlib.repmat(stadval,np.size(x,0),1)+np.matlib.repmat(meanval,np.size(x,0),1);

plt.figure(7)
plt.plot(t,var1,'b',label='Var 1',LineWidth=0.5)
plt.plot(t,var2,'g',label='Var 2',LineWidth=0.5)
plt.scatter(t,reconstr[:,0],c='r',s=8,label='Reconstructed')
plt.scatter(t,reconstr[:,1],c='r',s=8)
plt.xlabel('Time [s]')


# Residuals

Residuals1 = var1-reconstr[:,0];
Residuals2 = var2-reconstr[:,1];

plt.figure(8)
plt.subplot(2,2,1)
plt.plot(t,Residuals1,'b')
plt.xlabel('Time')
plt.ylabel('Residuals Var. 1')
plt.subplot(2,2,3)
plt.plot(t,Residuals1,'b')
plt.xlabel('Time')
plt.ylabel('Residuals Var. 2')
plt.subplot(2,2,2)
plt.hist(Residuals1)
plt.subplot(2,2,4)
plt.hist(Residuals2)
