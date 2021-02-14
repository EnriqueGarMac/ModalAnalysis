# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 20:47:15 2021

@author: Enrique GM
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

x = np.array([1,3,4,5,2,8,6,7])
N = len(x)
Nmed = int(N/2)

FFTr = fft(x)

plt.figure(1)
# Single-sided spectrum
Fs = 1;
Mod = np.abs(FFTr)
Po = Mod[0:Nmed]
f = np.arange(0,Nmed)*(Fs/N)
plt.plot(f,Po)
plt.xlabel('f')
plt.ylabel('Abs(f(t))')
plt.savefig('Una_hoja.jpg',dpi=300)

plt.figure(2)
# Double-sided spectrum
f =  np.arange(-Nmed,Nmed+1)*(Fs/N)
Pt = np.concatenate((Mod[Nmed:],Mod[0:Nmed+1]))
plt.plot(f,Pt)
plt.xlabel('f')
plt.ylabel('Abs(f(t))')
plt.savefig('Dos_hojas.jpg',dpi=300)