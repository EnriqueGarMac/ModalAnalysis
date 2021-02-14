# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 21:16:11 2021

@author: Enrique GM
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

Fs = 50

# Amplitude of the harmonics
A1 = 5
A2 = 2
A3 = 3

# Frequencies
f1 = 2
f2 = 7
f3 = 14

# Number of data samples
N = 2**10
Nmed = int(N/2)

# Time vector
dt = 1/Fs
t = np.arange(0,(N)*dt,dt)

# Harmonics
harmonic1 = A1*np.sin(2*np.pi*f1*t+2)
harmonic2 = A2*np.sin(2*np.pi*f2*t+1)
harmonic3 = A3*np.sin(2*np.pi*f3*t+5)

signal = harmonic1+harmonic2+harmonic3

plt.figure(1)
ax1=plt.subplot(4, 1, 1)
plt.plot(t,harmonic1,linewidth=1)
plt.ylabel('Harmonic 1')
ax1=plt.subplot(4, 1, 2)
plt.plot(t,harmonic2,linewidth=1)
plt.ylabel('Harmonic 2')
ax1=plt.subplot(4, 1, 3)
plt.plot(t,harmonic3,linewidth=1)
plt.ylabel('Harmonic 3')
ax1=plt.subplot(4, 1, 4)
plt.plot(t,signal,linewidth=1)
plt.xlabel('Time [s]')
plt.ylabel('Signal')
plt.tight_layout()
plt.savefig('Senales_harmonicas.jpg',dpi=300)

FFT = fft(signal)

plt.figure(2)
# Single-sided spectrum
Mod = np.abs(FFT)/N      # If I want to get the amplitude of the harmonics
# correcting for the total energy in the time-domain signal
P1 = Mod[0:Nmed]
P1[1:] = 2*P1[1:]
f = np.arange(0,Nmed)*(Fs/N)
df = f[1]-f[0]
plt.plot(f,P1,linewidth=4)
plt.title('Single-sided spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('|FFT|')
plt.savefig('Harmo_1hoja.jpg',dpi=300)


plt.figure(3)
plt.plot(np.abs(FFT))

plt.figure(4)
Mod = np.abs(FFT)/N      # If I want to get the amplitude of the harmonics
# Double-sided spectrum
f = np.arange(-Nmed,Nmed+1)*(Fs/N)
P2 = np.concatenate((Mod[Nmed:],Mod[0:Nmed+1]))
#P2[0:Nmed-1] = 2*P2[0:Nmed-1]
#P2[Nmed+2:] = 2*P2[Nmed+2:]
plt.plot(f,P2,linewidth=4)
plt.title('Double-sided spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('|FFT|')
plt.savefig('Harmo_2hojas.jpg',dpi=300)


# Check Parseval's Theorem

energy_signal = np.dot(signal,np.conj(signal))
energy_FFT = np.abs(np.dot(FFT,np.conj(FFT)) * (1/(N)))

print(energy_signal)
print(energy_FFT)