# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:41:39 2021

@author: Enrique GM
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
import time

dt = 0.5
shift = 20
Tt = 100
t = np.arange(0,Tt+dt,dt)
N = len(t)

signal1 = np.zeros((1,len(t)));
signal2 = np.zeros((1,len(t)));

signal1[0,49:] = np.exp(-0.1*t[49:])*np.cos(1.2*np.pi*1*t[49:])
signal2[0,49+shift:] = np.exp(-0.1*t[49+shift:])*np.cos(1.2*np.pi*1*t[49+shift:])


plt.figure(1)
plt.subplot(3,1,1)
plt.plot(t,signal1.T,'-b')
plt.subplot(3,1,2)
plt.plot(t,signal2.T,'-r')
plt.subplot(3,1,3)
c = np.correlate(signal1[0,:],signal2[0,:], "full")
lags = np.arange(-N+1,N,1)
plt.plot(dt*lags,c)



ax1 = plt.subplot(3,1,1)
ax2 = plt.subplot(3,1,2)
ax3 = plt.subplot(3,1,3)
lagserie = np.zeros([1,2*N])
corr = np.zeros([1,2*N])
for i in np.arange(1,np.size(signal1)+1,1):
   lag = i-1
   ax1.cla() 
   ax1.plot(t,signal1.T,'-b')
   
   ax2.cla() 
   ax2.plot(t,signal2.T,'--r')
   if lag == 0:
       ax2.plot(t[0:],signal2[0,lag:].T,'-r')
       corr[0,i-1] = np.sum(signal2*signal1)
   else:
       ax2.plot(t[0:-lag],signal2[0,lag:].T,'-r')
       corr[0,i-1] = np.sum(signal2[0,lag:]*signal1[0,0:-lag])
   ax3.cla() 
   lagserie[0,i-1] = -lag
   ax3.stem(lagserie.T,corr.T,'-b')
   plt.show()
   plt.pause(0.00001)

   

for i in np.arange(1,np.size(signal1)+1,1):
   lag = i;
   ax1.cla() 
   ax1.plot(t,signal1.T,'-b')
   
   ax2.cla() 
   ax2.plot(t,signal2.T,'--r')
   ax2.plot(t[lag+1:],signal2[0,1:-lag],'-r')

   ax3.cla() 
   lagserie[0,np.size(signal1)+i-1] = lag
   corr[0,np.size(signal1)+i-1] = np.sum(signal2[0,1:-lag]*signal1[0,lag+1:])
   ax3.stem(lagserie.T,corr.T,'-b')
   plt.show()
   plt.pause(0.00001)

plt.figure(3)
ax1=plt.subplot(3, 1, 1)
plt.plot(t,signal1.T,'-b')
ax2=plt.subplot(3, 1, 2)
plt.plot(t,signal2.T,'-r')
ax3=plt.subplot(3, 1, 3)
c = np.correlate(signal1[0,:],signal2[0,:], "full")
lags = np.arange(-N+1,N,1)
plt.stem(lags,c)
sort_index = np.argsort(lagserie) 
plt.plot(lagserie[0,sort_index].T,corr[0,sort_index].T,'-b')

