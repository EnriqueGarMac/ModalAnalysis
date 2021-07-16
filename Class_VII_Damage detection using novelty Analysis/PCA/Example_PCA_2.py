# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:50:47 2021

@author: Enrique GM
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from numpy.linalg import eig

def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days = 366)
    return day + dayfrac

# Read results

filename="Resonant_frequencies.txt"
expfreq = np.loadtxt(filename)

filename="Temperature.txt"
all_temp_TH = np.loadtxt(filename)

filename="Dates.txt"
Dates = np.loadtxt(filename)
Datespy = [matlab2datetime(tval) for tval in Dates]

plt.figure(1)
ax = plt.gca()
for i in np.arange(0,6+1,1):
   plt.scatter(Datespy,expfreq[:,i], s=2)

formatter = mdates.DateFormatter("%d-%m-%Y")
ax.xaxis.set_major_formatter(formatter)
plt.ylabel('Resonant frequency [Hz]')

plt.figure(2)
ax = plt.gca()
for i in np.arange(0,2,1):
   plt.scatter(Datespy,all_temp_TH[:,i], s=2)

formatter = mdates.DateFormatter("%d-%m-%Y")
ax.xaxis.set_major_formatter(formatter)
plt.ylabel('Temperature [Celsius degrees]')

# PCA

Y = expfreq

# Normalization
x = Y
meanval = np.mean(x,axis=0);
stadval = np.std(x,axis=0);
x = (x-np.matlib.repmat(meanval,np.size(x,0),1))/np.matlib.repmat(stadval,np.size(x,0),1);
COV = np.cov(x.T)
D, V = eig(COV)

sort_index = np.flip(np.argsort(D))

latent2 = D[sort_index];
coeff2 = V[:,sort_index];
D = latent2;
explainedvar = 100*D/np.sum(D);
score2 = np.matmul(x,coeff2)

plt.figure(3);
plt.bar([1,2,3,4,5,6,7],explainedvar)
plt.xticks([1,2,3,4,5,6,7], ('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'))
plt.ylabel('Explained variance [%]')

ll = 2
Z_vect=np.matmul(x,coeff2)
score2=Z_vect[:,np.arange(0,ll,1)];
reconstr=np.matmul(score2,coeff2[:,np.arange(0,ll,1)].T)
Yapprox = reconstr*np.matlib.repmat(stadval,np.size(x,0),1)+np.matlib.repmat(meanval,np.size(x,0),1);


plt.figure(4)
ax = plt.gca()
for i in np.arange(0,6+1,1):
   plt.scatter(Datespy,expfreq[:,i], s=2, c='b')
   plt.scatter(Datespy,Yapprox[:,i], s=2, c='r')

formatter = mdates.DateFormatter("%d-%m-%Y")
ax.xaxis.set_major_formatter(formatter)
plt.ylabel('Resonant frequency [Hz]')


# Residuals

R = Yapprox-Y

plt.figure(5)
for i in np.arange(0,6+1,1):
   axx = plt.subplot(7,2,2*(i+1)-1);
   plt.scatter(Datespy,R[:,i], s=2, c='b')
   formatter = mdates.DateFormatter(" ")
   if i==6:
       formatter = mdates.DateFormatter("%d-%m-%Y")
   axx.xaxis.set_major_formatter(formatter)
   plt.ylabel('R = '+str(i+1))

for i in np.arange(0,6+1,1):
   axx = plt.subplot(7,2,2*(i+1))
   plt.hist(R[:,i],bins=40)
   plt.xlim([-0.3,0.5])
   if i<6:
       axx.get_xaxis().set_ticks([])