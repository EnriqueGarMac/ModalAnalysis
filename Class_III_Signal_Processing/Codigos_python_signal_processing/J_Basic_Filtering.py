# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 18:31:10 2021

@author: Enrique GM
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy import signal

Fs = 100
dt = 1/Fs
x = np.arange(0,50+dt,dt)
N = np.size(x);
y = 2*np.sin(2*np.pi*15*x)+1.5*np.sin(2*np.pi*40*x)+x*0.2;
X = 2*(np.random.randn(N,1));
X[2000:,0] = X[2000:,0]+55-(0.002*np.arange(0,N-2000,1))**3;
signo = y+X[:,0]

plt.figure(1)
plt.plot(x,signo)
plt.xlabel('Time [s]')
plt.ylabel('f(t)')
plt.savefig('signal2filter.jpg',dpi=300)


# Fourier transform
signaltot = signo[0:2**12];
N = np.size(signaltot)
Nmed = int(N/2)

plt.figure(2)
FFT = fft(signaltot*np.hanning(len(signaltot)))
Mod = np.abs(FFT)/N      # If I want to get the amplitude of the harmonics
P1 = Mod[0:Nmed+1]
P1[1:] = 2*P1[1:]
f = np.linspace(0,Fs/2,Nmed+1, endpoint=True)
df = f[1]-f[0]
plt.plot(f,P1,linewidth=1)
plt.title('Single-sided spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('|FFT|')
plt.yscale('log')
plt.savefig('fftsignal2filter.jpg',dpi=300)



##########################################################
##########################################################
#####  MANUAL FILTERING
##########################################################
##########################################################



# Low-pass filter
FFT = fft(signaltot)

filterd = np.zeros((1,len(f)))
cutoff = 25
b = f>=cutoff 
pos = np.array(range(len(b)))
pos = pos[b]
filterd[0,0:pos[0]-1] = 1

filterd = np.concatenate((filterd[0,:],np.flip(np.conj(filterd[0,1:-1]))), axis=0)
FFT_filtered = FFT*filterd
filteredsignal = ifft(FFT_filtered)

plt.figure(3)
plt.subplot(2,1,1)
plt.plot(x[0:2**12],signaltot,'b')
plt.plot(x[0:2**12],filteredsignal,'r')
plt.ylabel('Signal')
plt.subplot(2,1,2)
FFT = fft(filteredsignal*np.hanning(len(signaltot)))
Mod = np.abs(FFT)/N       # If I want to get the amplitude of the harmonics
P1 = Mod[0:Nmed+1]
P1[1:] = 2*P1[1:]
f = np.linspace(0,Fs/2,Nmed+1, endpoint=True)
df = f[1]-f[0]
plt.plot(f,P1,linewidth=1)
plt.xlabel('Frequency [Hz]')
plt.ylabel('|FFT|')
plt.yscale('log')
plt.title('Low-pass filter')
plt.tight_layout()
plt.savefig('Low_pass_manual.jpg',dpi=300)


# High-pass filter
FFT = fft(signaltot) 

filterd = np.zeros((1,len(f)))
cutoff_f = 25;
b = f>=cutoff 
pos = np.array(range(len(b)))
pos = pos[b]
filterd[0,pos[0]:] = 1

filterd = np.concatenate((filterd[0,:],np.flip(np.conj(filterd[0,1:-1]))), axis=0);
FFT_filtered = FFT*filterd
filteredsignal = ifft(FFT_filtered)

plt.figure(4)
plt.subplot(2,1,1)
plt.plot(x[0:2**12],signaltot,'b')
plt.plot(x[0:2**12],filteredsignal,'r')
plt.ylabel('Signal')
plt.subplot(2,1,2)
FFT = fft(filteredsignal*np.hanning(len(signaltot)))
Mod = np.abs(FFT)/N       # If I want to get the amplitude of the harmonics
P1 = Mod[0:Nmed+1]
P1[1:] = 2*P1[1:]
f = np.linspace(0,Fs/2,Nmed+1, endpoint=True)
df = f[1]-f[0]
plt.plot(f,P1,linewidth=1)
plt.xlabel('Frequency [Hz]')
plt.ylabel('|FFT|')
plt.yscale('log')
plt.title('High-pass filter')
plt.tight_layout()
plt.savefig('High_pass_manual.jpg',dpi=300)

# Band-pass filter
FFT = fft(signaltot) 

filterd = np.ones((1,len(f)))
cutoff_fa = 5;
cutoff_fb = 25;
b = f>=cutoff_fb 
pos = np.array(range(len(b)))
pos = pos[b]
filterd[0,pos[0]:] = 0
b = f<=cutoff_fa 
pos = np.array(range(len(b)))
pos = pos[b]
filterd[0,0:pos[-1]] = 0

filterd = np.concatenate((filterd[0,:],np.flip(np.conj(filterd[0,1:-1]))), axis=0);
FFT_filtered = FFT*filterd
filteredsignal = ifft(FFT_filtered)

plt.figure(5)
plt.subplot(2,1,1)
plt.plot(x[0:2**12],signaltot,'b')
plt.plot(x[0:2**12],filteredsignal,'r')
plt.ylabel('Signal')
plt.subplot(2,1,2)
FFT = fft(filteredsignal*np.hanning(len(signaltot)))
Mod = np.abs(FFT)/N       # If I want to get the amplitude of the harmonics
P1 = Mod[0:Nmed+1]
P1[1:] = 2*P1[1:]
f = np.linspace(0,Fs/2,Nmed+1, endpoint=True)
df = f[1]-f[0]
plt.plot(f,P1,linewidth=1)
plt.xlabel('Frequency [Hz]')
plt.ylabel('|FFT|')
plt.yscale('log')
plt.title('Band-pass filter')
plt.tight_layout()
plt.savefig('Band_pass_manual.jpg',dpi=300)

##########################################################
##########################################################
#####  BUTTERWORTH FILTERING
##########################################################
##########################################################

fc = 25

plt.figure(6)
for ij in np.arange(1,5+1,1):
   b, a = signal.butter(ij, fc/(Fs/2), 'low')
   w, histbutter  = signal.freqz(b,a)
   plt.plot(Fs*w/(2*np.pi),np.abs(histbutter),linewidth=1, label='n='+str(ij))
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Filter Amplitude')
plt.title('Low-pass filter')
plt.savefig('Butter_low.jpg',dpi=300)

plt.figure(7)
for ij in np.arange(1,5+1,1):
   [b,a] = signal.butter(ij,fc/(Fs/2),'high');
   w, histbutter  = signal.freqz(b,a)
   plt.plot(Fs*w/(2*np.pi),np.abs(histbutter),linewidth=1, label='n='+str(ij))
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Filter Amplitude')
plt.title('High-pass filter')
plt.savefig('Butter_high.jpg',dpi=300)

plt.figure(8)
fc1 = 5
fc2 = 25
for ij in np.arange(1,5+1,1):
   [b,a] = signal.butter(ij,np.array([fc1,fc2])/(Fs/2),'bandpass');
   w, histbutter  = signal.freqz(b,a)
   plt.plot(Fs*w/(2*np.pi),abs(histbutter),linewidth=1, label='n='+str(ij))
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Filter Amplitude')
plt.title('Band-pass filter')
plt.savefig('Butter_band.jpg',dpi=300)

# ..............................................................
# Filter

plt.figure(9)
cutoff_f = 25;
[b,a] = signal.butter(1,cutoff_f/(Fs/2),'low');
filteredsignal = signal.lfilter(b,a,signaltot.T)

plt.subplot(2,1,1)
plt.plot(x[0:2**12],signaltot,'b')
plt.plot(x[0:2**12],filteredsignal,'r')
plt.ylabel('Signal')

plt.subplot(2,1,2)
del FFT
FFT = fft(filteredsignal*np.hanning(len(filteredsignal)))
Mod = np.abs(FFT)/N       # If I want to get the amplitude of the harmonics
P1 = Mod[0:Nmed+1]
P1[1:] = 2*P1[1:]
del FFT
FFT = fft(signaltot*np.hanning(len(signaltot)));  
Mod = np.abs(FFT)/N       # If I want to get the amplitude of the harmonics
P1o = Mod[0:Nmed+1]
P1o[1:] = 2*P1o[1:]
f = np.linspace(0,Fs/2,Nmed+1, endpoint=True)
df = f[1]-f[0]

plt.plot(f,P1o,'--b',linewidth=1)
plt.plot(f,P1,'r',linewidth=1)

plt.xlabel('Frequency [Hz]')
plt.ylabel('|FFT|')
plt.yscale('log')
plt.title('Low-pass filter')
plt.tight_layout()
plt.savefig('Band_pass_butter.jpg',dpi=300)



plt.figure(10)
cutoff_f = 25;
[b,a] = signal.butter(1,cutoff_f/(Fs/2),'high');
filteredsignal = signal.lfilter(b,a,signaltot.T)

plt.subplot(2,1,1)
plt.plot(x[0:2**12],signaltot,'b')
plt.plot(x[0:2**12],filteredsignal,'r')
plt.ylabel('Signal')

plt.subplot(2,1,2)
FFT = fft(filteredsignal*np.hanning(len(filteredsignal)))
Mod = np.abs(FFT)/N       # If I want to get the amplitude of the harmonics
P1 = Mod[0:Nmed+1]
P1[1:] = 2*P1[1:]
FFT = fft(signaltot*np.hanning(len(signaltot)))
Mod = np.abs(FFT)/N      # If I want to get the amplitude of the harmonics

plt.plot(f,P1o,'--b',linewidth=1)
plt.plot(f,P1,'r',linewidth=1)
plt.xlabel('Frequency [Hz]')
plt.ylabel('|FFT|')
plt.yscale('log')
plt.title('High-pass filter')
plt.tight_layout()
plt.savefig('High_pass_butter.jpg',dpi=300)


plt.figure(11)
fc1 = 5
fc2 = 25
[b,a] = signal.butter(1,np.array([fc1,fc2])/(Fs/2),'bandpass');
filteredsignal = signal.lfilter(b,a,signaltot.T)

plt.subplot(2,1,1)
plt.plot(x[0:2**12],signaltot,'b')
plt.plot(x[0:2**12],filteredsignal,'r')
plt.ylabel('Signal')

plt.subplot(2,1,2)
FFT = fft(filteredsignal*np.hanning(len(filteredsignal)))
Mod = np.abs(FFT)/N       # If I want to get the amplitude of the harmonics
P1 = Mod[0:Nmed+1]
P1[1:] = 2*P1[1:]
FFT = fft(signaltot*np.hanning(len(signaltot)))
Mod = np.abs(FFT)/N      # If I want to get the amplitude of the harmonics

plt.plot(f,P1o,'--b',linewidth=1)
plt.plot(f,P1,'r',linewidth=1)
plt.xlabel('Frequency [Hz]')
plt.ylabel('|FFT|')
plt.yscale('log')
plt.title('High-pass filter')