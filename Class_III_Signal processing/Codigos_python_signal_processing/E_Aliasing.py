# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:11:21 2021

@author: Enrique GM
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# Number of cycles
ncycles = 3 

# Frequency of the signal
fsignal = 5  # Hz
Tsignal = 1/fsignal
wsignal = 2*np.pi*fsignal

# Time
t = np.arange(0,ncycles*Tsignal,0.0005)
# Signal
signal = 2*np.cos(2*np.pi*fsignal*t+3)

# Plot
plt.figure(1)
plt.plot(t,signal)


# Sampling

fsampling = 6
dt = 1/fsampling
ts =  np.arange(0,ncycles*Tsignal,dt)

sampledsignal = 2*np.cos(2*np.pi*fsignal*ts+3);

# Plot
plt.figure(2)
plt.plot(t,signal,'b')
plt.plot(ts,sampledsignal,'ro')



# 
l = 1;
signal2 = 2*np.cos(2*np.pi*(fsignal+l*fsampling)*t+3)
l = -1;
signal3 = 2*np.cos(2*np.pi*(fsignal+l*fsampling)*t+3)

plt.figure(3)
plt.plot(t,signal,'b')
plt.plot(t,signal2,'--k')
plt.plot(t,signal3,'-.g')
plt.plot(ts,sampledsignal,'ro')
plt.legend()
plt.savefig('aliasing_ex2.jpg',dpi=300)


