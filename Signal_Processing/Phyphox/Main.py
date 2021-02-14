# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 13:16:57 2021

@author: Enrique GM
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import welch, periodogram
from scipy import interpolate
from scipy.signal import find_peaks

# READ DATA

plt.figure(1)
T = pd.read_csv('Enrique.csv')
time = np.array(T.iloc[:,[0]])
accx = np.array(T.iloc[:,[1]])
accy = np.array(T.iloc[:,[2]])
accz = np.array(T.iloc[:,[3]])
accx = accx-np.mean(accx)
accy = accy-np.mean(accy)
accz = accz-np.mean(accz)
dtm = np.mean(np.diff(time.T))
tnew = np.arange(0, np.max(time.T), dtm)
tnew = tnew.T
fcubic = interpolate.interp1d(time[:,0], accx[:,0], kind='cubic')
accx = fcubic(tnew)
fcubic = interpolate.interp1d(time[:,0], accy[:,0], kind='cubic')
accy = fcubic(tnew)
fcubic = interpolate.interp1d(time[:,0], accz[:,0], kind='cubic')
accz = fcubic(tnew)
plt.plot(tnew,accx,label='Acc_x')
plt.plot(tnew,accy,label='Acc_y')
plt.plot(tnew,accz,label='Acc_z')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s$^2$]')
plt.legend()
plt.savefig('time_series.jpg',dpi=300)
y = np.vstack((accx,accy,accz)).T
dt = tnew[1]-tnew[0]
Fs = 1/dt

elimini = int(2/dt)
elimend = 200

plt.figure(2)
y = np.delete(y, np.arange(0,elimini,1).astype(int), 0)
tnew = np.delete(tnew, np.arange(0,elimini,1).astype(int), 0)
y = np.delete(y, np.arange(-1,-elimend,-1).astype(int), 0)
tnew = np.delete(tnew, np.arange(-1,-elimend,-1).astype(int), 0)
plt.plot(tnew,y)

# PSD

[fp,pxx] = welch(y,Fs,nperseg=2**13,return_onesided=1,scaling='density', detrend=False, axis = 0);
channel = 2

plt.figure(3)
plt.plot(fp,10*np.log10(pxx[:,0]),label='Acc_x')
plt.plot(fp,10*np.log10(pxx[:,1]),label='Acc_y')
plt.plot(fp,10*np.log10(pxx[:,2]),label='Acc_z')
posmax = np.argmax(pxx[:,channel])
plt.plot(fp[posmax], 10*np.log10(pxx[posmax,channel]), "x")
plt.xlim([0,Fs/2])
plt.title('Welchs method')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.tight_layout()
plt.savefig('Welch_Phyphox.jpg',dpi=300)

print('Peak frequency: '+str(fp[posmax])+' Hz')