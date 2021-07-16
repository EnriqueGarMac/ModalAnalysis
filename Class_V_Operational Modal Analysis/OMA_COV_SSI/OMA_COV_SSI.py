'''
Created on 4. aug. 2018

@author: Ole-Martin
'''



import sys
from matplotlib import pyplot as plt
import numpy as np
from OMA_functions import *
from stabplot import stabplot
from matplotlib.animation import FuncAnimation
import warnings
import pip

try:
    __import__('mplcursors')
except ImportError:
    pip.main(['install', 'mplcursors']) 
        
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
# ..........................................................................
# ..........................................................................
# READ SIGNAL
# ..........................................................................
# ..........................................................................


signals = np.genfromtxt('Random_Test_200Hz.csv', delimiter=',')

plt.figure(1)
plt.plot(signals)


# ..........................................................................
# ..........................................................................
# COV-SSI
# ..........................................................................
# ..........................................................................

fs = 200


# Cov-SSI settings
i = 20
s = 4
orders = np.arange(2, 70+2, 2)
stabcrit = {'freq':0.2, 'damping': 0.2, 'mac': 0.3}


# Find poles by Cov-SSI
lambd, phi = covssi(signals, fs, i, orders) 
# Find stable poles
lambd_stab, phi_stab, orders_stab, ix_stab = find_stable_poles(lambd, phi, orders, s, stabcrit=stabcrit, indicator='mac') # Stable poles
# Clustering
[fn,zeta,phi,Clustersize] = Clustering(lambd_stab, phi_stab)

ind = np.argsort(fn)
fn = fn[ind]
zeta = zeta[ind]
Clustersize = Clustersize[ind]
phi = phi[:,ind]

# Plot stabilization diagram
nperseg = 2**12
zp = 4
nfft = nperseg*zp
f, Pxx = welch(signals[:,0], fs, 'hanning', nperseg=nperseg, nfft=nfft)
fig = stabplot(lambd, orders, lambd_stab, orders_stab, fn, psd_freq=f, psd_y=Pxx, frequency_unit='Hz', freq_range=[0,fs/2], damped_freq=False)
plt.savefig('COV-SSI.jpg',dpi=300)


# ..........................................................................
# ..........................................................................
# VALIDATION ANALYSIS
# ..........................................................................
# ..........................................................................


print('Resonant frequencies')
print(fn)
print('Damping ratios')
print(zeta)

MOD = complex_to_normal_mode(phi)

lh = 4 # Width
lv = 3 # Height
fig = plt.figure(1,figsize=(20,5))
plotframemodes(1, MOD, lh, lv, fn)


###############################################################################################################################################
mplot = 1
mult = 1

xun = np.array([0,0,0,0,0,lh,lh,lh,lh,lh,np.nan,0,lh,np.nan,0,lh,np.nan,0,lh,np.nan,0,lh])
yun = np.array([0,lv,2*lv,3*lv,4*lv,4*lv,3*lv,2*lv,lv,0,np.nan,lv,lv,np.nan,2*lv,2*lv,np.nan,3*lv,3*lv,np.nan,4*lv,4*lv])
xde,yde = convert_mode(mult*MOD[:,mplot-1], lh, lv)
x_data = xde
y_data = yun 
    
fig, ax = plt.subplots()
ax.set_xlim(-4, 10)
ax.plot(xun,yun,'--k')
plt.title('Modo ' + str(mplot) + ' - Frequency : '+str(round(float(fn[mplot-1]), 2))+' Hz')
line, = ax.plot(x_data, y_data,'-bo',linewidth=3)

def animation_frame(i,MOD,mplot,lh,lv,mult, fn):
	x_data,yde = convert_mode(MOD[:,mplot-1]*np.cos(i*2*np.pi/15)*mult, lh, lv)
	line.set_xdata(x_data)
	return line, 

animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, 15, 00.1), fargs = (MOD,mplot,lh,lv,mult, fn), interval=10)
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
#plt.savefig('MAC_matrix.jpg',dpi=300)