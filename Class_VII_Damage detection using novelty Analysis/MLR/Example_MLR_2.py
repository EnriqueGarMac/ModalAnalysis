# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:37:02 2021

@author: Enrique GM
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

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


# MLR

Y = expfreq
X = np.concatenate((np.ones((np.size(all_temp_TH,0),1)),all_temp_TH),axis=1)
XXT = np.matmul(X.T,X)
iXXT = np.linalg.pinv(XXT)
beta = np.matmul(np.matmul(iXXT,X.T),Y)
Yapprox = np.matmul(X,beta)

plt.figure(3)
ax = plt.gca()
for i in np.arange(0,6+1,1):
   plt.scatter(Datespy,expfreq[:,i], s=2, c='b')
   plt.scatter(Datespy,Yapprox[:,i], s=2, c='r')

formatter = mdates.DateFormatter("%d-%m-%Y")
ax.xaxis.set_major_formatter(formatter)
plt.ylabel('Resonant frequency [Hz]')


# Residuals

R = Yapprox-Y

plt.figure(4)
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
