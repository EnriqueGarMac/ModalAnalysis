# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 21:08:03 2021

@author: Enrique GM
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import welch
from scipy.signal import chirp
from scipy.signal import spectrogram
from mpl_toolkits.mplot3d import Axes3D


Fs = 1000
dt = 1/Fs
Tt = 2
t = np.linspace(0,Tt,int(Tt/dt), endpoint=True)
N = np.size(t)
f1 = 25
f2 = 50
t2 = 1
y = chirp(t,f1,t2,f2,'quadratic')

plt.figure(1)
plt.plot(t,y)
plt.savefig('signalsweep.jpg',dpi=300)

nseg = 2**7
fS, tS, PSD = spectrogram(y, fs=Fs, window='hanning',
                                      nperseg=nseg, noverlap=nseg/3,
                                      detrend=False, scaling='spectrum')

Samp = 10*np.log10(PSD)
plt.figure(2)
plt.pcolormesh(tS, fS, Samp, cmap='viridis', shading='gouraud')
plt.colorbar()
plt.ylabel('Frequency [kHz]')
plt.axis('tight')
plt.xlabel('Time [s]');
plt.savefig('timefreq1.jpg',dpi=300)


fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(fS[:, None], tS[None, :], Samp, cmap='viridis')
ax.set_xlabel('Frequency [Hz]', fontsize=10, rotation=150)
ax.set_ylabel('Time [s]', fontsize=10)
ax.set_zlabel('Power/Frequency [dB/Hz]', fontsize=10, rotation=60)
plt.show()
plt.savefig('timefreq2.jpg',dpi=300)
