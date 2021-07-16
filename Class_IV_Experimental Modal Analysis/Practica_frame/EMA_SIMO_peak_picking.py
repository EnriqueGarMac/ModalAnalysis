# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:46:43 2021

@author: Enrique GM
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from numpy.linalg import eig
import scipy.sparse.linalg as sla
from scipy.signal import find_peaks
import pandas as pd
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from EMA_functions import *


###############################################################################################################################################
# Leemos las señales

data = pd.read_csv('Test_disp200Hz.csv', sep=',',header=None)
data = data.values
Fs = 200
N = int(2**np.floor(np.log(np.size(data,0))/np.log(2)))
force = data[0:N,0]
displacements = data[0:N,1:5]

plt.figure(1)
plt.plot(displacements[:,0])
plt.plot(displacements[:,1])
plt.plot(displacements[:,2])
plt.plot(displacements[:,3])

###############################################################################################################################################
# Valores teóricos para comparar

TFreq = pd.read_csv('Theoretical_frequencies.csv', sep=',',header=None)
TFreq = TFreq.values
TMS = pd.read_csv('Theoretical_Mode_shapes.csv', sep=',',header=None)
TMS = TMS.values

###############################################################################################################################################

# FRFs
                           
# SIMO
# Una sola entrada y varias salidas
k = 3
nfig = 3
MODexp = np.zeros((4,4))
MODexp = MODexp.astype(complex)

# Vector con las frecuencias que buscamos
searchfreqs = TFreq*(2*np.pi)

for lserie in np.arange(3,-1,-1):
    
    Force = fft(force)
    resp = fft(displacements[:,lserie])
    N = np.size(force)
    Nmed = int(N/2)
    Hlk = resp[0:Nmed]/Force[0:Nmed]
    f = np.linspace(0,Fs/2,Nmed, endpoint=True)
    wserie = 2*np.pi*f
    yhat = savgol_filter(np.abs(Hlk), 21, 3)
    totpeaks, _ = find_peaks(yhat, height=0)
    #if lserie==3:
    if lserie>-999:    
        peaks = (searchfreqs*0).astype(int)
        for i in np.arange(0,len(searchfreqs),1):
            idx,_ = find_nearest(wserie[totpeaks], searchfreqs[i])
            peaks[i] = totpeaks[idx]
    freqpeak = wserie[peaks]
    amppeak = np.abs(Hlk)[peaks]
    
    
    print(freqpeak)
    
    
    plt.figure(nfig)
    plt.subplot(1,2,1)
    plt.plot(wserie,np.abs(Force[0:Nmed]),'k')
    plt.subplot(1,2,2)
    nfig = nfig+1
    plt.plot(wserie,np.abs(Hlk),'k')
    plt.plot(wserie[peaks], np.abs(Hlk)[peaks], "x")
    plt.stem(wserie[peaks],np.abs(Hlk)[peaks], '--k')
    damp = np.zeros((len(freqpeak),1))
    for j in np.arange(0,len(freqpeak),1):
        valsearch = amppeak[j]/np.sqrt(2)
        fsearch = 5*2*np.pi
        nps = int(fsearch/(wserie[1]-wserie[0]))
        searchband = np.arange(peaks[j]-nps,peaks[j])
        ampband = np.abs(Hlk)[searchband]
        freqband = wserie[searchband]
        idx,val1 = find_nearest(ampband,valsearch)
        f1 = freqband[idx]
        searchband = np.arange(peaks[j],peaks[j]+nps)
        ampband = np.abs(Hlk)[searchband]
        freqband = wserie[searchband]
        idx,val2 = find_nearest(ampband,valsearch)
        f2 = freqband[idx]
        damp[j] = (f2-f1)/(f2+f1)
        plt.plot(np.array([f1,f2]),np.array([val1,val2]),'r')
    
    plt.show()
    plt.xlim((0,500))
    plt.ylim((0.1*np.min(np.abs(Hlk)),10*np.max(np.abs(Hlk))))
    plt.ylabel('$Abs(H_{lk}(\omega))$')
    plt.xlabel('$\omega$')
    plt.yscale('log')
    plt.tight_layout()
    #print(damp)
    
    
    seta1exp = damp[0]
    seta2exp = damp[1]
    seta3exp = damp[2]
    seta4exp = damp[3]
    wiexp = freqpeak
    H1exp = 1/(wiexp[0]**2-wserie**2+1j*2*seta1exp*wiexp[0]*wserie)
    H2exp = 1/(wiexp[1]**2-wserie**2+1j*2*seta2exp*wiexp[1]*wserie)
    H3exp = 1/(wiexp[2]**2-wserie**2+1j*2*seta3exp*wiexp[2]*wserie)
    H4exp = 1/(wiexp[3]**2-wserie**2+1j*2*seta4exp*wiexp[3]*wserie)
    
    Hll = Hlk[peaks]
    H1cmp = H1exp[peaks]
    H2cmp = H2exp[peaks]
    H3cmp = H3exp[peaks]
    H4cmp = H4exp[peaks]
    Hcmp = np.column_stack((np.column_stack((np.column_stack((H1cmp,H2cmp)),H3cmp)),H4cmp))
    MM = np.linalg.pinv(Hcmp)
    MM = np.matmul(MM,Hll)
    if lserie==3:
        MODexp[lserie,:] = np.sqrt(MM)[:,0]
    else:
        MODexp[lserie,:] = MM[:,0]/MODexp[k,:]   
        
        
###############################################################################################################################################
Expfreq = freqpeak/(2*np.pi)
print('Damping')
print(damp)
print('Resonant frequencies')
print(Expfreq)

MODexpreal = complex_to_normal_mode(MODexp, max_dof=50, long=True)

lh = 4
lv = 3
fig = plt.figure(nfig+1,figsize=(20,5))
plotframemodes(5, MODexpreal, TMS, lh, lv, TFreq, Expfreq)

   
###############################################################################################################################################
#######################################       MAC matrix test    ###############################################################


MACvals = MAC(MODexp, MODexp)
fig = plt.figure(nfig+2)  
ax1 = fig.add_subplot(111, projection='3d')
top = MACvals.reshape((np.size(MACvals), 1))
x = np.array([np.arange(1,np.shape(MODexpreal)[1]+1),]*np.shape(MODexpreal)[1]).reshape((np.size(top), 1))-0.5
y = np.array([np.arange(1,np.shape(MODexpreal)[1]+1),]*np.shape(MODexpreal)[1]).transpose().reshape((np.size(top), 1))-0.5
bottom = np.zeros_like(top)
width = depth = 1
cmap = plt.cm.get_cmap('jet') # Get desired colormap - you can change this!
max_height = np.max(top)   # get range of colorbars so we can normalize
min_height = np.min(top)
# scale each z to [0,1], and get their rgb values
rgba = [cmap((k-min_height)/max_height) for k in top[:,0]] 
ax1.bar3d(x[:,0], y[:,0], bottom[0,:], width, depth, top[:,0], shade=True, color=rgba)
ax1.set_title('MAC matrix')
ax1.set_ylabel('Experimental mode shapes')
ax1.set_xlabel('Experimental mode shapes')
ax1.set_title('Experimental vs Experimental')
plt.savefig('MAC_matrix.jpg',dpi=300)


MACvals = MAC(TMS, MODexp)
fig = plt.figure(nfig+3)  
ax1 = fig.add_subplot(111, projection='3d')
top = MACvals.reshape((np.size(MACvals), 1))
x = np.array([np.arange(1,np.shape(MODexpreal)[1]+1),]*np.shape(MODexpreal)[1]).reshape((np.size(top), 1))-0.5
y = np.array([np.arange(1,np.shape(MODexpreal)[1]+1),]*np.shape(MODexpreal)[1]).transpose().reshape((np.size(top), 1))-0.5
bottom = np.zeros_like(top)
width = depth = 1
cmap = plt.cm.get_cmap('jet') # Get desired colormap - you can change this!
max_height = np.max(top)   # get range of colorbars so we can normalize
min_height = np.min(top)
# scale each z to [0,1], and get their rgb values
rgba = [cmap((k-min_height)/max_height) for k in top[:,0]] 
ax1.bar3d(x[:,0], y[:,0], bottom[0,:], width, depth, top[:,0], shade=True, color=rgba)
ax1.set_title('MAC matrix')
ax1.set_ylabel('Theoretical mode shapes')
ax1.set_xlabel('Experimental mode shapes')
ax1.set_title('Experimental vs Theoretical')
plt.savefig('MAC_matrix.jpg',dpi=300)
