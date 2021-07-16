# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:14:48 2021

@author: Enrique GM
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d

ncycles = 10 # Number of cycles
fsignal = 5  # Hz
Tsignal = 1/fsignal


t = np.linspace(0,ncycles*Tsignal,int(ncycles*Tsignal/0.001), endpoint=True)

# Real analog signal
real_signal = 2*np.sin(t*2*np.pi*fsignal+2)

plt.figure(1)
plt.plot(t,real_signal,'b',linewidth=2)

# Time-discrete signal

facquired = 6
dtacquired = 1/facquired
tacquired = np.linspace(0,ncycles*Tsignal,int(ncycles*Tsignal/dtacquired), endpoint=True)
acquired_signal = 2*np.sin(tacquired*2*np.pi*fsignal+2)


plt.figure(2)
plt.plot(t,real_signal,'b',linewidth=2,label='real signal')
plt.plot(tacquired,acquired_signal,'or',linewidth=2,label='sampling points')
plt.legend()
plt.savefig('aliasing_ex1.jpg',dpi=300)

f = interp1d(tacquired, acquired_signal, kind='cubic')
acquired_signalr = f(t)

plt.figure(3)
plt.plot(t,real_signal,'b',linewidth=2)
plt.plot(tacquired,acquired_signal,'or',linewidth=2)
plt.plot(t,acquired_signalr,'--r',linewidth=2)

plt.savefig('aliasing_ex3.jpg',dpi=300)

# Frequency analysis


Fsr = 1/0.001;
N = len(real_signal)
Nmed = int(N/2)
df = ((Fsr/2))/((N-1)/2)
Yr = fft(real_signal)
Yr = Yr[0:Nmed]
fr = np.arange(0,Nmed)*(Fsr/N)

N = len(acquired_signalr)
df = ((Fsr/2))/((N-1)/2)
Ys = fft(acquired_signalr)
Ys = Ys[0:Nmed]
fs = np.arange(0,Nmed)*(Fsr/N)

plt.figure(4)
plt.plot(fr,abs(Yr),'b',linewidth=2,label='real')
plt.plot(fs,abs(Ys),'--r',linewidth=2,label='subsampled')
plt.xlim([0,10])
plt.legend()
plt.savefig('aliasing_ex4.jpg',dpi=300)


