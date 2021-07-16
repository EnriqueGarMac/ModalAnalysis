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
from scipy import signal



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
T = 120
N = int(T/dt)
tv = np.linspace(0,T,N,endpoint=True)
forcv = np.zeros((N,4))
forcv[:,0] = 2*np.random.rand(N)
forcv[:,1] = 1*np.random.rand(N)
forcv[:,2] = 3*np.random.rand(N)
forcv[:,3] = 0.5*np.random.rand(N)
B2 = np.eye(4)
uo = np.zeros((1,4))
vo = np.zeros((1,4))
[displacements,velocities,accelerations,tf] = state_space_solver_comp_dis(K,C,M,forcv,B2,uo,vo,dt)

plt.figure(2,figsize=(13,8))
plt.subplot(4,1,1)
plt.plot(tf,forcv[1:,0])
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
plt.savefig('Random_measurements.jpg',dpi=300)

test = np.vstack((accelerations[:,0],accelerations[:,1],accelerations[:,2],accelerations[:,3])).T
name = 'Random_Test_'+str(Fs)+'Hz.csv'
np.savetxt(name, test, delimiter=',')