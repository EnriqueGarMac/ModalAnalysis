# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:26:11 2021

@author: Enrique GM
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d

Fs = 60

# Amplitude of the harmonic
A1 = 5
# Frequencies
f1 = 2


# Number of data samples
N = 2**12
Nmed = int(N/2)

# Time vector
dt = 1/Fs
t = np.linspace(0,(N-1)*dt,N, endpoint=True)

# Harmonic
signal = A1*np.sin(2*np.pi*f1*t+2)


plt.figure(1)
ax1=plt.subplot(3, 1, 1)
plt.plot(t,signal,linewidth=2)
plt.ylabel('Signal')
ax2=plt.subplot(3, 1, 2)
plt.plot(t,np.hanning(len(signal)),linewidth=2)
plt.plot(t,np.sin(np.pi*t/t[-1])**2,'--r',linewidth=2)
plt.ylabel('Window')
ax3=plt.subplot(3, 1, 3)
plt.plot(t,signal*np.hanning(len(signal)),linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Windowed Signal')
plt.savefig('windowhann.jpg',dpi=300)


plt.figure(2)
# Single-sided spectrum
FFT = fft(signal);  
Mod = np.abs(FFT)/N      # If I want to get the amplitude of the harmonics
P1 = Mod[0:Nmed]
P1[1:] = 2*P1[1:]
FFT_wind = fft(signal*np.hanning(len(signal)));  
Mod_wind = abs(FFT_wind)/N;      # If I want to get the amplitude of the harmonics
P1_wind = Mod_wind[0:Nmed]
P1_wind[1:] = 2*P1_wind[1:]
f = np.arange(0,Nmed)*(Fs/N)
df = f[1]-f[0];

plt.plot(f,P1,'b',linewidth=4,label='original')
plt.plot(f,P1_wind,'r',linewidth=4,label='windowed')
plt.legend()
plt.title('Single-sided spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('|FFT|')
plt.savefig('freq_leak.jpg',dpi=300)
 

