# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:44:14 2021

@author: Enrique GM
"""

from OMA_functions import *
import scipy.sparse.linalg as sla
from scipy.linalg import eig
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.signal import welch
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import welch
from matplotlib.animation import FuncAnimation



k1 = 7.5*10**7
k2 = k1
k3 = k1
k4 = k1
m1 = 3600
m2 = 2850
m3 = m2
m4 = 1800

damping = 1/100

kelem = [k1,k2,k3,k4]
M = np.array([[m1,0,0,0],[0,m2,0,0],[0,0,m3,0],[0,0,0,m4]])
K = np.array([[kelem[0]+kelem[1],-kelem[1],0,0],
              [-kelem[1],kelem[1]+kelem[2],-kelem[2],0],
              [0,-kelem[2],kelem[2]+kelem[3],-kelem[3]],
              [0,0,-kelem[3],kelem[3]]])


# ..........................................................................
# ..........................................................................
# MODAL ANALYSIS
# ..........................................................................
# ..........................................................................

[L,MOD]=eigen2(K,M,np.array([]),5);
wrad = np.sqrt(L)
frequenciesHz=np.sqrt(L)/(2*np.pi);

print('Resonant frequencies')
print(frequenciesHz)


lh = 4 # Width
lv = 3 # Height
fig = plt.figure(1,figsize=(20,5))
plotframemodes(1, MOD, lh, lv, frequenciesHz)



###############################################################################################################################################
mplot = 1
mult = 55

xun = np.array([0,0,0,0,0,lh,lh,lh,lh,lh,np.nan,0,lh,np.nan,0,lh,np.nan,0,lh,np.nan,0,lh])
yun = np.array([0,lv,2*lv,3*lv,4*lv,4*lv,3*lv,2*lv,lv,0,np.nan,lv,lv,np.nan,2*lv,2*lv,np.nan,3*lv,3*lv,np.nan,4*lv,4*lv])
xde,yde = convert_mode(mult*MOD[:,mplot-1], lh, lv)
x_data = xde
y_data = yun 
    
fig, ax = plt.subplots()
ax.set_xlim(-4, 10)
ax.plot(xun,yun,'--k')
plt.title('Modo ' + str(mplot) + ' - Frequency : '+str(round(float(frequenciesHz[mplot-1]), 2))+' Hz')
line, = ax.plot(x_data, y_data,'-bo',linewidth=3)

def animation_frame(i,MOD,mplot,lh,lv,mult, frequenciesHz):
	x_data,yde = convert_mode(MOD[:,mplot-1]*np.cos(i*2*np.pi/15)*mult, lh, lv)
	line.set_xdata(x_data)
	return line, 

animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, 15, 00.1), fargs = (MOD,mplot,lh,lv,mult, frequenciesHz), interval=10)
plt.show()
animation.save('mode_shape_'+str(mplot)+'.gif', writer='imagemagick', fps=30)
