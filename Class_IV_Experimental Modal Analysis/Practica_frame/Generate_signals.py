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

name = 'Theoretical_frequencies.csv'
np.savetxt(name, frequenciesHz, delimiter=',')

name = 'Theoretical_Mode_shapes.csv'
np.savetxt(name, MOD, delimiter=',')


[alpha0,alpha1] = Rayleighdamping(damping,damping,wrad[0],wrad[1])
C = alpha1*K+alpha0*M

# ..........................................................................
# ..........................................................................
# DYNAMIC ANALYSIS
# ..........................................................................
# ..........................................................................

Fs = 200
dt = 1/Fs
T = 40
N = int(T/dt)
wex = 15*2*np.pi
tv = np.linspace(0,T,N,endpoint=True)
forcv = np.zeros((N,1))
forcv[994:995,0] = 1
B2 = np.array([[0],[0],[0],[1.]])
uo = np.zeros((1,4))
vo = np.zeros((1,4))
[displacements,velocities,accelerations,tf] = state_space_solver_comp_dis(K,C,M,forcv,B2,uo,vo,dt)



plt.figure(2)
plt.subplot(3,1,1)
for ij in np.array([0,1,2,3]): 
    plt.plot(tf,displacements[:,ij],label = 'Level  '+"{:.2f}".format(int(ij)))
plt.xlabel('Time [s]')
plt.title('Displacements [m]')
plt.legend()
plt.subplot(3,1,2)
for ij in np.array([0,1,2,3]): 
    plt.plot(tf,velocities[:,ij],label = 'Level  '+"{:.2f}".format(int(ij)))
plt.xlabel('Time [s]')
plt.title('Velocities [m/s$^2$]')
plt.legend()
plt.subplot(3,1,3)
for ij in np.array([0,1,2,3]): 
    plt.plot(tf,accelerations[:,ij],label = 'Level  '+"{:.2f}".format(int(ij)))
plt.xlabel('Time [s]')
plt.title('Accelerations [m/s$^2$]')
plt.legend()
plt.tight_layout()
plt.show



test = np.vstack((forcv[0:-1,0],accelerations[:,0],accelerations[:,1],accelerations[:,2],accelerations[:,3])).T
name = 'Test_'+str(Fs)+'Hz.csv'
np.savetxt(name, test, delimiter=',')

test = np.vstack((forcv[0:-1,0],displacements[:,0],displacements[:,1],displacements[:,2],displacements[:,3])).T
name = 'Test_disp'+str(Fs)+'Hz.csv'
np.savetxt(name, test, delimiter=',')

# ..........................................................................
# ..........................................................................
# MODAL ANALYSIS
# ..........................................................................
# ..........................................................................


A,B = statespace_discrete(K, C, M, B2, dt)


[mu,X1] = eig(A)

D = np.log(mu)/dt

frequenciesHz = np.abs(D)/(2*np.pi)
MOD = X1[0:4,:]

plt.figure(3)
plt.subplot(2,4,1)
for i in np.array([0,1,2,3]):
     plt.arrow(0,0,np.real(MOD[i,0]),np.imag(MOD[i,0]),color='b',width=1e-5,head_width = 6e-5, head_length = 0.5e-3)
plt.axis('equal')
plt.title('Freq.: '+"{:.2f}".format(float(frequenciesHz[0]))+ 'Hz')
plt.subplot(2,4,2)
for i in np.array([0,1,2,3]):
     plt.arrow(0,0,np.real(MOD[i,1]),np.imag(MOD[i,1]),color='b',width=1e-5,head_width = 6e-5, head_length = 0.5e-3)
plt.axis('equal')
plt.title('Freq.: '+"{:.2f}".format(float(frequenciesHz[1]))+ 'Hz')
plt.subplot(2,4,3)
for i in np.array([0,1,2,3]):
     plt.arrow(0,0,np.real(MOD[i,2]),np.imag(MOD[i,2]),color='b',width=1e-5,head_width = 6e-5, head_length = 0.5e-3)
plt.axis('equal')
plt.title('Freq.: '+"{:.2f}".format(float(frequenciesHz[2]))+ 'Hz')
plt.subplot(2,4,4)
for i in np.array([0,1,2,3]):
     plt.arrow(0,0,np.real(MOD[i,3]),np.imag(MOD[i,3]),color='b',width=1e-5,head_width = 6e-5, head_length = 0.5e-3)
plt.axis('equal')
plt.title('Freq.: '+"{:.2f}".format(float(frequenciesHz[3]))+ 'Hz')
plt.subplot(2,4,5)
for i in np.array([0,1,2,3]):
     plt.arrow(0,0,np.real(MOD[i,4]),np.imag(MOD[i,4]),color='b',width=1e-5,head_width = 6e-5, head_length = 0.5e-3)
plt.axis('equal')
plt.title('Freq.: '+"{:.2f}".format(float(frequenciesHz[4]))+ 'Hz')
plt.subplot(2,4,6)
for i in np.array([0,1,2,3]):
     plt.arrow(0,0,np.real(MOD[i,5]),np.imag(MOD[i,5]),color='b',width=1e-5,head_width = 6e-5, head_length = 0.5e-3)
plt.axis('equal')
plt.title('Freq.: '+"{:.2f}".format(float(frequenciesHz[5]))+ 'Hz')
plt.subplot(2,4,7)
for i in np.array([0,1,2,3]):
     plt.arrow(0,0,np.real(MOD[i,6]),np.imag(MOD[i,6]),color='b',width=1e-5,head_width = 6e-5, head_length = 0.5e-3)
plt.axis('equal')
plt.title('Freq.: '+"{:.2f}".format(float(frequenciesHz[6]))+ 'Hz')
plt.subplot(2,4,8)
for i in np.array([0,1,2,3]):
     plt.arrow(0,0,np.real(MOD[i,7]),np.imag(MOD[i,7]),color='b',width=1e-5,head_width = 6e-5, head_length = 0.5e-3)
plt.axis('equal')
plt.title('Freq.: '+"{:.2f}".format(float(frequenciesHz[7]))+ 'Hz')
plt.tight_layout()
plt.show

