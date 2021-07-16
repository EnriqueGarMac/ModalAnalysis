# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:23:11 2021

@author: Enrique GM
"""
from scipy.fft import fft, fftfreq, fftshift
import pytest
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import pyEMA
from EMA_functions import *


###############################################################################################################################################
# Valores teóricos para comparar

TFreq = pd.read_csv('Theoretical_frequencies.csv', sep=',',header=None)
TFreq = TFreq.values
TMS = pd.read_csv('Theoretical_Mode_shapes.csv', sep=',',header=None)
TMS = TMS.values


###############################################################################################################################################
# Leemos las señales

data = pd.read_csv('Test_200Hz.csv', sep=',',header=None)
data = data.values
Fs = 200
N = int(2**np.floor(np.log(np.size(data,0))/np.log(2)))
force = data[0:N,0]
accel = data[0:N,1:5]
dt = 1/Fs
tvector = np.linspace(0,(N-1)*dt,N)


fig = plt.figure(25)
plt.subplot(1,2,1)
plt.plot(tvector,force)
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.subplot(1,2,2)
plt.plot(tvector,accel[:,0],label='level 1')
plt.plot(tvector,accel[:,1],label='level 2')
plt.plot(tvector,accel[:,2],label='level 3')
plt.plot(tvector,accel[:,3],label='level 4')
plt.xlabel('Time [s]')
plt.ylabel(r"Acceleration [m/s$^2$]")
plt.legend()
plt.tight_layout()
plt.show()
    
###############################################################################################################################################


N = np.size(force)
Nmed = int(N/2)
freq = np.linspace(0,Fs/2,Nmed,endpoint=True)
FRF = np.zeros((4,Nmed))
FRF = FRF.astype(dtype=complex)

for lserie in np.arange(3,-1,-1):
    
    Force = fft(force)
    resp = fft(accel[:,lserie])
    Hlk = resp[0:Nmed]/Force[0:Nmed]
    FRF[lserie,:] = Hlk[0:Nmed]


acc = pyEMA.Model(frf=FRF, freq=freq, lower=5, 
                upper=Fs/2, pol_order_high=60)

acc.get_poles(show_progress=False)
acc.select_poles()
H, A = acc.get_constants()
u, s, vh = np.linalg.svd(A, full_matrices=True)
Expfreq = acc.nat_freq
MOD = complex_to_normal_mode(A, max_dof=50, long=True)

###############################################################################################################################################


TFreq = pd.read_csv('Theoretical_frequencies.csv', sep=',',header=None)
TFreq = TFreq.values
TMS = pd.read_csv('Theoretical_Mode_shapes.csv', sep=',',header=None)
TMS = TMS.values

###############################################################################################################################################

lh = 4
lv = 3
fig = plt.figure(3,figsize=(20,5))
plotframemodes(5, MOD, TMS, lh, lv, TFreq, Expfreq)

###############################################################################################################################################
mplot = 3
mult = 2

xun = np.array([0,0,0,0,0,lh,lh,lh,lh,lh,np.nan,0,lh,np.nan,0,lh,np.nan,0,lh,np.nan,0,lh])
yun = np.array([0,lv,2*lv,3*lv,4*lv,4*lv,3*lv,2*lv,lv,0,np.nan,lv,lv,np.nan,2*lv,2*lv,np.nan,3*lv,3*lv,np.nan,4*lv,4*lv])
xde,yde = convert_mode(mult*MOD[:,mplot-1], lh, lv)
x_data = xde
y_data = yun 
    
fig, ax = plt.subplots()
ax.set_xlim(-4, 10)
ax.plot(xun,yun,'--k')
plt.title('Modo ' + str(mplot) + ' - Frequency :'+str(round(float(TFreq[mplot-1]), 2))+' Hz')
line, = ax.plot(x_data, y_data,'-bo',linewidth=3)

def animation_frame(i,MOD,mplot,lh,lv,mult, TFreq, Expfreq):
	x_data,yde = convert_mode(MOD[:,mplot-1]*np.cos(i*2*np.pi/15)*mult, lh, lv)
	line.set_xdata(x_data)
	return line, 

animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, 15, 00.1), fargs = (MOD,mplot,lh,lv,mult, TFreq, Expfreq), interval=10)
plt.show()
animation.save('mode_shape.gif', writer='imagemagick', fps=30)



###############################################################################################################################################
#######################################       MAC matrix test    ###############################################################


MACvals = MAC(A, A)
fig = plt.figure(200)  
ax1 = fig.add_subplot(111, projection='3d')
top = MACvals.reshape((np.size(MACvals), 1))
x = np.array([np.arange(1,np.shape(MOD)[1]+1),]*np.shape(MOD)[1]).reshape((np.size(top), 1))-0.5
y = np.array([np.arange(1,np.shape(MOD)[1]+1),]*np.shape(MOD)[1]).transpose().reshape((np.size(top), 1))-0.5
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


MACvals = MAC(TMS, MOD)
fig = plt.figure(201)  
ax1 = fig.add_subplot(111, projection='3d')
top = MACvals.reshape((np.size(MACvals), 1))
x = np.array([np.arange(1,np.shape(MOD)[1]+1),]*np.shape(MOD)[1]).reshape((np.size(top), 1))-0.5
y = np.array([np.arange(1,np.shape(MOD)[1]+1),]*np.shape(MOD)[1]).transpose().reshape((np.size(top), 1))-0.5
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
