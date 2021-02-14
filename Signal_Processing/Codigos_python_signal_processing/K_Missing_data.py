# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 19:57:33 2021

@author: Enrique GM
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x = np.arange(0,1+0.01,0.01)
N = len(x)
y = 4+2*np.sin(2*np.pi*5*x)+0.1*np.random.randn(1,N)
y = y[0,:]

ngaps = 20;
p = np.random.permutation(N);
y[p[0:ngaps]] = np.nan

plt.figure(1)
plt.plot(x,y[:],linewidth=2)
plt.title('Incomplete signal')
plt.savefig('Incomplete_signal.jpg',dpi=300)

# Fill missing data

# Methods

# piecewise linear interpolation 
# previous
# spline
# Other methods
#‘linear’
#‘nearest’
#‘nearest-up’
#‘zero’
#‘slinear’
#‘quadratic’
#‘cubic’
#‘previous’
#‘next’

plt.figure(2)
# piecewise linear interpolation 
plt.subplot(3,1,1)
plt.plot(x,y[:],'--ob',linewidth=2)
bi = np.isnan(y[:])
comp = np.empty_like(y) 
comp[:] = y 
f = interp1d(x[~bi], y[~bi],kind='linear',fill_value="extrapolate")
comp[bi]= f(x[bi])
plt.plot(x,comp,'--r')
plt.plot(x[bi],comp[bi],'ro')
plt.title('linear interpolation')
plt.subplot(3,1,2)
plt.plot(x,y[:],'--ob',linewidth=2)
bi = np.isnan(y[:])
comp = np.empty_like(y) 
comp[:] = y 
f = interp1d(x[~bi], y[~bi],kind='previous',fill_value="extrapolate")
comp[bi]= f(x[bi])
plt.plot(x,comp,'--r')
plt.plot(x[bi],comp[bi],'ro')
plt.title('previous interpolation')
plt.subplot(3,1,3)
plt.plot(x,y[:],'--ob',linewidth=2)
bi = np.isnan(y[:])
comp = np.empty_like(y) 
comp[:] = y 
f = interp1d(x[~bi], y[~bi],kind='cubic',fill_value="extrapolate")
comp[bi]= f(x[bi])
plt.plot(x,comp,'--r')
plt.plot(x[bi],comp[bi],'ro')
plt.title('cubic interpolation')
plt.tight_layout()
plt.savefig('Missing_data_comp.jpg',dpi=300)