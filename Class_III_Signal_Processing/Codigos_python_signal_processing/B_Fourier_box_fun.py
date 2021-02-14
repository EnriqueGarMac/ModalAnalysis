# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 20:05:17 2021

@author: Enrique GM
Example - Fourier transform of box function
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

T = 1 # Width

# Time
t = np.linspace(-20,20,2**12);
dt = t[1]-t[0] # time step
# Box-function
b = np.zeros((len(t),1))
npp = int(np.round(0.5*T/dt))
b[0:npp] = 1
b[-npp+1:] = 1


plt.figure(1)
plt.plot(t,b,linewidth=2)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.savefig('Square_signal_TS.jpg',dpi=300)

# Analytic Fourier Transform
df = 0.001
f = np.arange(-1/(2*dt),1/(2*dt),df)
bf = T*np.sinc(f*T)   # ojjo! Python define sincx como función sinc normalizada: sinc(x) = sinc(pi*x)/(oi*x)

plt.figure(2)
plt.plot(f,bf)


# Using FFT
FFTr=fft(b.T).T
N = int(len(FFTr))
FFTr = FFTr*dt
df2 = 1/(N*dt)
FFTr = np.concatenate((FFTr[int(N/2)+1:-1], FFTr[0:int(N/2)]), axis=None)
freqpos = np.concatenate((-np.flip(np.arange(df2,(N/2)*df2,df2)), np.arange(df2,(N/2)*df2,df2)), axis=None)


plt.figure(3)
plt.plot(f,np.real(bf),'b',label = 'Analitica')
plt.plot(freqpos,np.real(FFTr),'.r',label = 'Numérica')
plt.xlabel('f')
plt.ylabel('X(f)')
plt.legend()
plt.savefig('Square_signal_FT.jpg',dpi=300)
