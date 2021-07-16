'''
Created on 4. aug. 2018

@author: Ole-Martin
'''



import sys
from matplotlib import pyplot as plt
import numpy as np
from OMA_functions import *
from matplotlib.animation import FuncAnimation


signals = np.genfromtxt('Measurement_1_cleaned.txt', delimiter=',')
samplingFreq = 180 # Sampling Frequency [Hz]
lh = 0.3 # Width
lv = 0.3 # Height

plt.figure(1)
plt.plot(signals)





PSD_matrix,s1,ms,Frequencies = mainFDD(signals,samplingFreq,peakThresh=-300, nfft = 2**12)
# Lanzar solo hasta aquí si se desea seleccionar los picos con el ratón

potfrequencies = np.array([4.6,13.7,22.1])

Frequencies, dbs1, ResFreq, chosenPeaksMag, phi = FDDpeak(PSD_matrix,s1,ms,Frequencies,potfrequencies,peakThresh=-300)
plt.savefig('FDD_results.jpg',dpi=300)

damping =  FDDdamping(PSD_matrix,s1,ms,Frequencies,samplingFreq,ResFreq,phi,deltafreq=2)



# ..........................................................................
# ..........................................................................
# VALIDATION ANALYSIS
# ..........................................................................
# ..........................................................................


print('Resonant frequencies [Hz]')
print(ResFreq)
print('Damping ratios [%]')
print(damping)

MOD = complex_to_normal_mode(phi)
fig = plt.figure(1,figsize=(10,5))
plotframemodes(0.1, MOD, lh, lv, ResFreq)
plt.savefig('Id_Mode_shapes.jpg',dpi=300)

#########################################################################################################################################
# ANIMATION
#########################################################################################################################################
mplot = 3   # Mode to plot
mult = 0.04   # Amplification factor

xun = np.array([0,0,0,0,lh,lh,lh,lh,np.NaN,0,lh,np.NaN,0,lh,np.NaN,0,lh])
yun = np.array([0,lv,2*lv,3*lv,3*lv,2*lv,1*lv,0,np.NaN,lv,lv,np.NaN,2*lv,2*lv,np.NaN,3*lv,3*lv])
xde,yde = convert_mode(mult*MOD[:,mplot-1], lh, lv)
x_data = xde
y_data = yun 
    
fig, ax = plt.subplots()
ax.set_xlim(-0.1, 0.4)
ax.plot(xun,yun,'--k')
plt.title('Modo ' + str(mplot) + ' - Frequency : '+str(round(float(ResFreq[mplot-1]), 2))+' Hz')
line, = ax.plot(x_data, y_data,'-bo',linewidth=3)

def animation_frame(i,MOD,mplot,lh,lv,mult, ResFreq):
	x_data,yde = convert_mode(MOD[:,mplot-1]*np.cos(i*2*np.pi/15)*mult, lh, lv)
	line.set_xdata(x_data)
	return line, 

animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, 15, 00.1), fargs = (MOD,mplot,lh,lv,mult, ResFreq), interval=10)
plt.show()
animation.save('mode_shape_'+str(mplot)+'.gif', writer='imagemagick', fps=30)


###############################################################################################################################################


MACvals = MAC(phi, phi)
fig = plt.figure(201)  
ax1 = fig.add_subplot(111, projection='3d')
top = MACvals.reshape((np.size(MACvals), 1))
x = np.array([np.arange(1,np.shape(phi)[1]+1),]*np.shape(phi)[1]).reshape((np.size(top), 1))-0.5
y = np.array([np.arange(1,np.shape(phi)[1]+1),]*np.shape(phi)[1]).transpose().reshape((np.size(top), 1))-0.5
bottom = np.zeros_like(top)
width = depth = 1
cmap = plt.cm.get_cmap('jet') # Get desired colormap - you can change this!
max_height = np.max(top)   # get range of colorbars so we can normalize
min_height = np.min(top)
# scale each z to [0,1], and get their rgb values
rgba = [cmap((k-min_height)/max_height) for k in top[:,0]] 
ax1.bar3d(x[:,0], y[:,0], bottom[0,:], width, depth, top[:,0], shade=True, color=rgba)
ax1.set_title('MAC matrix')
plt.savefig('MAC_matrix.jpg',dpi=300)