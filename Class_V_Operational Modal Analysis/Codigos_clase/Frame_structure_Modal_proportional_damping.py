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


b = 4
h = 2
x = np.array([0,0,0,0,0,b,b,b,b,b,np.NaN,0,b,np.NaN,0,b,np.NaN,0,b,np.NaN,0,b])
y = np.array([0,h,2*h,3*h,4*h,4*h,3*h,2*h,1*h,0,np.NaN,h,h,np.NaN,2*h,2*h,np.NaN,3*h,3*h,np.NaN,4*h,4*h])

plt.figure(1,figsize=(16, 8))
plt.subplot(1,4,1)
xd = x.copy()
mult = 60
mode = 0
xd[1] = xd[1]+mult*MOD[0,mode]
xd[2] = xd[2]+mult*MOD[1,mode]
xd[3] = xd[3]+mult*MOD[2,mode]
xd[4] = xd[4]+mult*MOD[3,mode]
xd[5] = xd[5]+mult*MOD[3,mode]
xd[6] = xd[6]+mult*MOD[2,mode]
xd[7] = xd[7]+mult*MOD[1,mode]
xd[8] = xd[8]+mult*MOD[0,mode]
xd[11] = xd[11]+mult*MOD[0,mode]
xd[12] = xd[12]+mult*MOD[0,mode]
xd[14] = xd[14]+mult*MOD[1,mode]
xd[15] = xd[15]+mult*MOD[1,mode]
xd[17] = xd[17]+mult*MOD[2,mode]
xd[18] = xd[18]+mult*MOD[2,mode]
xd[20] = xd[20]+mult*MOD[3,mode]
xd[21] = xd[21]+mult*MOD[3,mode]
plt.plot(xd,y,'-b',Linewidth=4)
plt.plot(x,y,'--k',Linewidth=2)
plt.xlabel('x')
plt.title('Freq.: '+"{:.2f}".format(float(frequenciesHz[0]))+ 'Hz')
plt.axis('equal')
plt.subplot(1,4,2)
xd = x.copy()
mult = 60
mode = 1
xd[1] = xd[1]+mult*MOD[0,mode]
xd[2] = xd[2]+mult*MOD[1,mode]
xd[3] = xd[3]+mult*MOD[2,mode]
xd[4] = xd[4]+mult*MOD[3,mode]
xd[5] = xd[5]+mult*MOD[3,mode]
xd[6] = xd[6]+mult*MOD[2,mode]
xd[7] = xd[7]+mult*MOD[1,mode]
xd[8] = xd[8]+mult*MOD[0,mode]
xd[11] = xd[11]+mult*MOD[0,mode]
xd[12] = xd[12]+mult*MOD[0,mode]
xd[14] = xd[14]+mult*MOD[1,mode]
xd[15] = xd[15]+mult*MOD[1,mode]
xd[17] = xd[17]+mult*MOD[2,mode]
xd[18] = xd[18]+mult*MOD[2,mode]
xd[20] = xd[20]+mult*MOD[3,mode]
xd[21] = xd[21]+mult*MOD[3,mode]
plt.plot(xd,y,'-b',Linewidth=4)
plt.plot(x,y,'--k',Linewidth=2)
plt.xlabel('x')
plt.title('Freq.: '+"{:.2f}".format(float(frequenciesHz[1]))+ 'Hz')
plt.axis('equal')
plt.subplot(1,4,3)
xd = x.copy()
mult = 60
mode = 2
xd[1] = xd[1]+mult*MOD[0,mode]
xd[2] = xd[2]+mult*MOD[1,mode]
xd[3] = xd[3]+mult*MOD[2,mode]
xd[4] = xd[4]+mult*MOD[3,mode]
xd[5] = xd[5]+mult*MOD[3,mode]
xd[6] = xd[6]+mult*MOD[2,mode]
xd[7] = xd[7]+mult*MOD[1,mode]
xd[8] = xd[8]+mult*MOD[0,mode]
xd[11] = xd[11]+mult*MOD[0,mode]
xd[12] = xd[12]+mult*MOD[0,mode]
xd[14] = xd[14]+mult*MOD[1,mode]
xd[15] = xd[15]+mult*MOD[1,mode]
xd[17] = xd[17]+mult*MOD[2,mode]
xd[18] = xd[18]+mult*MOD[2,mode]
xd[20] = xd[20]+mult*MOD[3,mode]
xd[21] = xd[21]+mult*MOD[3,mode]
plt.plot(xd,y,'-b',Linewidth=4)
plt.plot(x,y,'--k',Linewidth=2)
plt.xlabel('x')
plt.title('Freq.: '+"{:.2f}".format(float(frequenciesHz[2]))+ 'Hz')
plt.axis('equal')
plt.subplot(1,4,4)
xd = x.copy()
mult = 60
mode = 3
xd[1] = xd[1]+mult*MOD[0,mode]
xd[2] = xd[2]+mult*MOD[1,mode]
xd[3] = xd[3]+mult*MOD[2,mode]
xd[4] = xd[4]+mult*MOD[3,mode]
xd[5] = xd[5]+mult*MOD[3,mode]
xd[6] = xd[6]+mult*MOD[2,mode]
xd[7] = xd[7]+mult*MOD[1,mode]
xd[8] = xd[8]+mult*MOD[0,mode]
xd[11] = xd[11]+mult*MOD[0,mode]
xd[12] = xd[12]+mult*MOD[0,mode]
xd[14] = xd[14]+mult*MOD[1,mode]
xd[15] = xd[15]+mult*MOD[1,mode]
xd[17] = xd[17]+mult*MOD[2,mode]
xd[18] = xd[18]+mult*MOD[2,mode]
xd[20] = xd[20]+mult*MOD[3,mode]
xd[21] = xd[21]+mult*MOD[3,mode]
plt.plot(xd,y,'-b',Linewidth=4)
plt.plot(x,y,'--k',Linewidth=2)
plt.xlabel('x')
plt.title('Freq.: '+"{:.2f}".format(float(frequenciesHz[3]))+ 'Hz')
plt.axis('equal')
plt.tight_layout()
plt.show
plt.savefig('Modal_Basic_Example.jpg',dpi=300)



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

masamodal = np.dot(np.dot(MOD.T,M),MOD)
rigidezmodal = np.dot(np.dot(MOD.T,K),MOD)
amortmodal = np.dot(np.dot(MOD.T,C),MOD)
B2 = np.zeros((4,4))
B2[3,3] = 1
fvec = np.dot(MOD.T,B2)

def f(u,x,fvec,tv,m,c,k):
     fint = interp1d(tv, fvec, kind='linear', fill_value="extrapolate")
     forceint= fint(x)
     return (u[1],-(c/m)*u[1]-(k/m)*u[0]+forceint)

moddisplacements = np.zeros((N,4))
modvelocities = moddisplacements.copy()
modaccelerations = np.zeros((N-1,4))
displacements = moddisplacements.copy()
velocities = moddisplacements.copy()
accelerations = np.zeros((N-1,4))


for ij in np.array([0,1,2,3]): 
  y0 = [0,0]
  vectff = np.dot(fvec,forcv.T)
  us = integrate.odeint(f,y0,tv,args=(vectff[ij,:],tv,masamodal[ij,ij],amortmodal[ij,ij],rigidezmodal[ij,ij]),rtol = 1e-6, atol=1e-8)
  moddisplacements[:,ij] = us[:,0]
  modvelocities[:,ij] = us[:,1]
  modaccelerations[:,ij] = np.diff(us[:,1], axis=0)/dt


displacements = np.dot(MOD,moddisplacements.T).T
velocities = np.dot(MOD,modvelocities.T).T
accelerations = np.dot(MOD,modaccelerations.T).T


plt.figure(2)
plt.subplot(3,1,1)
for ij in np.array([0,1,2,3]): 
    plt.plot(tv,displacements[:,ij],label = 'Level  '+"{:.2f}".format(int(ij)))
plt.xlabel('Time [s]')
plt.title('Displacements [m]')
plt.legend()
plt.subplot(3,1,2)
for ij in np.array([0,1,2,3]): 
    plt.plot(tv,velocities[:,ij],label = 'Level  '+"{:.2f}".format(int(ij)))
plt.xlabel('Time [s]')
plt.title('Velocities [m/s$^2$]')
plt.legend()
plt.subplot(3,1,3)
for ij in np.array([0,1,2,3]): 
    plt.plot(tv[0:-1],accelerations[:,ij],label = 'Level  '+"{:.2f}".format(int(ij)))
plt.xlabel('Time [s]')
plt.title('Accelerations [m/s$^2$]')
plt.legend()
plt.tight_layout()
plt.show
