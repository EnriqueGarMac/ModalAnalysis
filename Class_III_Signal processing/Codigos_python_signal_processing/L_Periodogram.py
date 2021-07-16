# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 20:36:57 2021

@author: Enrique GM
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import periodogram

Fs = 1000
dt = 1/Fs
N = 2**11
Nmed = int(N/2)
t = np.linspace(0,(N-1)*dt,N, endpoint=True)
x = np.cos(2*np.pi*100*t)+np.random.randn(1,len(t))
fxx, pxx = periodogram(x,Fs,nfft=np.size(x),window=np.ones(np.size(x)),return_onesided=1,scaling='spectrum', detrend=False)
pxx[0,0] = 2*pxx[0,0] 

FFT = fft(x);

# Single-sided spectrum
Mod = np.abs(FFT)**2/(Fs*N)
P1 = Mod[0,0:Nmed+1]
f = np.linspace(0,Fs/2,Nmed+1, endpoint=True)
df = f[1]-f[0]

plt.figure(1)
plt.plot(fxx,pxx[0,:],'--b')
plt.plot(f,P1,'--r')
plt.title('Single-sided spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD')