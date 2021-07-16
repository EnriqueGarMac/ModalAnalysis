# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:44:14 2021

@author: Enrique GM
"""

from OMA_functions import *
import scipy.sparse.linalg as sla
from scipy.linalg import eig
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

damping = 1/100

kelem = [k1,k2,k3,k4]
M = np.array([[m1,0,0,0],[0,m2,0,0],[0,0,m3,0],[0,0,0,m4]])
K = np.array([[kelem[0]+kelem[1],-kelem[1],0,0],
              [-kelem[1],kelem[1]+kelem[2],-kelem[2],0],
              [0,-kelem[2],kelem[2]+kelem[3],-kelem[3]],
              [0,0,-kelem[3],kelem[3]]])


# ..........................................................................
# ..........................................................................
# MODAL ANALYSIS
# ..........................................................................
# ..........................................................................

[L,MOD]=eigen2(K,M,np.array([]),5);
wrad = np.sqrt(L)
frequenciesHz=np.sqrt(L)/(2*np.pi);

[alpha0,alpha1] = Rayleighdamping(damping,damping,wrad[0],wrad[1])
C = alpha1*K+alpha0*M

# ..........................................................................
# ..........................................................................
# DYNAMIC ANALYSIS
# ..........................................................................
# ..........................................................................

Fs = 200
dt = 1/Fs
T = 10
N = int(T/dt)
wex = 15*2*np.pi
tv = np.linspace(0,T,N,endpoint=True)
forcv = np.zeros((N,4))
forcv[:,3] = 2*np.sin(wex*tv)
forcv[int(N/2):] = 0
B2 = np.zeros((4,4))
B2[3,3] = 1
uo = np.zeros((1,4))
vo = np.zeros((1,4))
[displacements,velocities,accelerations,tf] = state_space_solver_comp_dis(K,C,M,forcv,B2,uo,vo,dt)



plt.figure(2,figsize=(13,8))
plt.subplot(4,1,1)
plt.plot(tf,forcv[1:,3])
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.subplot(4,1,2)
for ij in np.array([0,1,2,3]): 
    plt.plot(tf,displacements[:,ij],label = 'Level  '+"{:.2f}".format(int(ij)))
plt.xlabel('Time [s]')
plt.ylabel('Displacements [m]')
plt.subplot(4,1,3)
for ij in np.array([0,1,2,3]): 
    plt.plot(tf,velocities[:,ij],label = 'Level  '+"{:.2f}".format(int(ij)))
plt.xlabel('Time [s]')
plt.ylabel('Velocities [m/s$^2$]')
plt.subplot(4,1,4)
for ij in np.array([0,1,2,3]): 
    plt.plot(tf,accelerations[:,ij],label = 'Level  '+"{:.2f}".format(int(ij)))
plt.xlabel('Time [s]')
plt.ylabel('Accelerations [m/s$^2$]')
plt.legend()
plt.tight_layout()
plt.show
plt.savefig('Response_state_space_discrete.jpg',dpi=300)


# ..........................................................................
# ..........................................................................
# MODAL ANALYSIS
# ..........................................................................
# ..........................................................................


A,B = statespace_discrete(K, C, M, B2, dt)


[mu,X1] = eig(A)

D = np.log(mu)/dt

frequenciesHzSP = np.abs(D)/(2*np.pi)
damp =  -100*np.real(D)/(2*np.pi*frequenciesHzSP)

X1 = X1[0:4,:]

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