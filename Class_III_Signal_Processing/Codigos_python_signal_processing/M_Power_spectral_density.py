# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 21:08:03 2021

@author: Enrique GM
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import welch

Fs = 20
N = 2**4
Nmed = int(N/2)

dt = 1/Fs
t = np.linspace(0,(N-1)*dt,N, endpoint=True)

signal = 2*np.sin(2*np.pi*5*t)+0*np.random.randn(1,N);


plt.figure(1)
plt.plot(t,signal[0,:])

FFT = fft(signal)  

plt.figure(2)
Mod = np.abs(FFT);
P1 = Mod[0,0:Nmed+1]
f = np.linspace(0,Fs/2,Nmed+1, endpoint=True)
df = f[1]-f[0]
plt.plot(f,P1,linewidth=2)
plt.title('Single-sided spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('|FFT|')



plt.figure(3)
[fp,pxx] = welch(signal,Fs,nfft=np.size(signal),window=np.ones(np.size(signal)),noverlap=0,return_onesided=1,scaling='density', detrend=False)
plt.plot(fp,pxx[0,:],'-ob',linewidth=2)
plt.plot(f,P1*2/Fs,'--or',linewidth=2)
plt.title('Single-sided spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')

np.sum(signal*np.conj(signal))
np.sum(FFT*np.conj(FFT)) * (1/(N))
np.sum(pxx*Fs)