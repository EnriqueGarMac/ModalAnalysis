# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:44:14 2021

@author: Enrique GM
"""

from OMA_functions import *
import scipy.sparse.linalg as sla
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.signal import welch
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import welch




k1 = 7.5*10**7
k2 = k1
k3 = k1
k4 = k1
m1 = 3600
m2 = 2850
m3 = m2
m4 = 1800
damping = 0.001;  # Damping ratio

kelem = [k1,k2,k3,k4]
M = np.array([[m1,0,0,0],[0,m2,0,0],[0,0,m3,0],[0,0,0,m4]])
K = np.array([[kelem[0]+kelem[1],-kelem[1],0,0],
              [-kelem[1],kelem[1]+kelem[2],-kelem[2],0],
              [0,-kelem[2],kelem[2]+kelem[3],-kelem[3]],
              [0,0,-kelem[3],kelem[3]]])


# MODAL ANALYSIS

# Undamped
[L,MOD]=eigen2(K,M,np.array([]),5);
wrad = np.sqrt(L)
frequenciesHz=np.sqrt(L)/(2*np.pi);

[alpha0,alpha1] = Rayleighdamping(damping,damping,wrad[0],wrad[1])
C = alpha1*K+alpha0*M


# Matriz de estado

ndofs = np.shape(K)[0]
A = np.zeros([2*ndofs, 2*ndofs])
A[0:ndofs, ndofs:2*ndofs] = np.eye(ndofs)
A[ndofs:2*ndofs, 0:ndofs] = -np.dot(np.linalg.inv(M), K)
A[ndofs:2*ndofs, ndofs:2*ndofs] = -np.dot(np.linalg.inv(M), C)
    
[D,X1]=sla.eigs(A,k=ndofs*2, M=None, which='SM')

X1 = X1[0:4,:]
frequenciesHzSP = np.abs(D)/(2*np.pi)
damp =  -100*np.real(D)/(2*np.pi*frequenciesHzSP)



MACvals = MAC(X1, MOD)
fig = plt.figure(201)  
ax1 = fig.add_subplot(111, projection='3d')
top = MACvals.reshape((np.size(MACvals), 1))
x = np.array([np.arange(1,np.shape(X1)[1]+1),]*np.shape(MOD)[1]).reshape((np.size(top), 1))-0.5
y = np.array([np.arange(1,np.shape(MOD)[1]+1),]*np.shape(X1)[1]).transpose().reshape((np.size(top), 1))-0.5
bottom = np.zeros_like(top)
width = depth = 1
cmap = plt.cm.get_cmap('jet') # Get desired colormap - you can change this!
max_height = np.max(top)   # get range of colorbars so we can normalize
min_height = np.min(top)
# scale each z to [0,1], and get their rgb values
rgba = [cmap((k-min_height)/max_height) for k in top[:,0]] 
ax1.bar3d(x[:,0], y[:,0], bottom[0,:], width, depth, top[:,0], shade=True, color=rgba)
ax1.set_title('MAC matrix')
ax1.set_ylabel('Theoretical mode shapes')
ax1.set_xlabel('Experimental mode shapes')
ax1.set_title('Experimental vs Theoretical')
plt.savefig('MAC_matrix.jpg',dpi=300)
