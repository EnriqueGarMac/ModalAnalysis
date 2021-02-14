# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 11:36:51 2021

@author: Enrique GM
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import welch, periodogram


filename="Ambient_Vibration_200Hz.txt"
y = np.loadtxt(filename)*9.81
N = np.size(y);

Fs = 200;
dt = 1/Fs
t = np.linspace(0,(N-1)*dt,N, endpoint=True)

plt.figure(1)
plt.plot(t,y)
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s$^2$]')
plt.savefig('Signal_bridge.jpg',dpi=300)


# PSD

[fp,pxx] = welch(y,Fs,nperseg=2**14,return_onesided=1,scaling='density', detrend=False);
[f2,pxx2] = periodogram(y,Fs,return_onesided=1,scaling='density', detrend=False);

plt.figure(2)
ax1 = plt.subplot(2,1,1);
plt.plot(f2,pxx2)
plt.xlim([0,Fs/2])
plt.title('Periodogram')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [(m/s$^2$)$^2$/Hz]')
plt.yscale('log')

ax2 = plt.subplot(2,1,2);
plt.plot(fp,pxx)
plt.xlim([0,Fs/2])
plt.title('Welchs method')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [(m/s$^2$)$^2$/Hz]')
plt.yscale('log')
plt.tight_layout()
plt.savefig('Preiodogram_vs_welch_bridge.jpg',dpi=300)