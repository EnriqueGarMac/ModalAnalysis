# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 14:30:11 2021

@author: Enrique GM
"""

# Code to generate clean measurements from AVT of experimental frame

import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
package = 'resampy'
import pip
def import_or_install(package):
    try:
        __import__(package)
    except:
        pip.main(['install', package]) 
        print('Installing '+package+'...')
import_or_install(package)
import resampy
from scipy.signal import welch, periodogram

name_input_file = 'Measurement_1.txt';
name_output_file = 'Measurement_1_cleaned.txt';

# Number of lines to delete from the beginning and the end of the measurements
dellines = 10
# Sampling frequency of the output signals
Fs_s = 180  

############################################################################

initialmeasurements = loadtxt(name_input_file, delimiter='\t', skiprows=0)

initialmeasurements = np.delete(initialmeasurements, np.arange(dellines), 0)
initialmeasurements = np.delete(initialmeasurements, -np.arange(dellines), 0)


initialmeasurements[:,0] = initialmeasurements[:,0]-initialmeasurements[0,0]
dts = np.diff(initialmeasurements[:,0])
dt = np.mean(np.abs(dts))
Fs = 1/dt
print('Sampling frequency: '+str(Fs)+' Hz')


newt = np.arange(0,initialmeasurements[-1,0]+1/Fs,1/Fs)
finalmeas = np.zeros((len(newt),4))

finalmeas[:,0]= newt
f = interp1d(initialmeasurements[:,0], initialmeasurements[:,1],kind='nearest',fill_value="extrapolate")
finalmeas[:,1]= f(newt)
f = interp1d(initialmeasurements[:,0], initialmeasurements[:,2],kind='nearest',fill_value="extrapolate")
finalmeas[:,2]= f(newt)
f = interp1d(initialmeasurements[:,0], initialmeasurements[:,3],kind='nearest',fill_value="extrapolate")
finalmeas[:,3]= f(newt)

y1 = resampy.resample(initialmeasurements[:,1], Fs, Fs_s)
y2 = resampy.resample(initialmeasurements[:,2], Fs, Fs_s)
y3 = resampy.resample(initialmeasurements[:,3], Fs, Fs_s)
tt = np.arange(0,(1/Fs_s)*(len(y1)),1/Fs_s)
data = np.column_stack((np.column_stack((y1,y2)),y3))

plt.figure(1)
plt.plot(tt,data[:,0],'r',linewidth=2,label = 'Level 1')
plt.plot(tt,data[:,1],'g',linewidth=2,label = 'Level 2')
plt.plot(tt,data[:,2],'b',linewidth=2,label = 'Level 3')
plt.legend()
plt.title('Original signals')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s2]')
plt.savefig('Raw_signal.jpg',dpi=300)

data = signal.detrend(data,axis=0)
plt.figure(2)
plt.plot(tt,data[:,0],'r',linewidth=2,label = 'Level 1')
plt.plot(tt,data[:,1],'g',linewidth=2,label = 'Level 2')
plt.plot(tt,data[:,2],'b',linewidth=2,label = 'Level 3')
plt.legend()
plt.title('Processed signals')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s2]')
plt.savefig('Processed_signal.jpg',dpi=300)

# PSD

[fp,pxx] = welch(data,Fs_s,nperseg=2**14,return_onesided=1,scaling='density', detrend=False, axis=0);

plt.figure(3)
plt.plot(fp,pxx)
plt.xlim([0,Fs_s/2])
plt.title('Welchs method')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [(m/s$^2$)$^2$/Hz]')
plt.yscale('log')
plt.tight_layout()
plt.savefig('Welch_frame.jpg',dpi=300)


np.savetxt(name_output_file, data, delimiter=',')