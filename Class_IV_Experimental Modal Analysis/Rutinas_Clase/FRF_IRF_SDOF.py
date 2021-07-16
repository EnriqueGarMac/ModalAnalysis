# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:43:41 2021

@author: Enrique GM
"""

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



m = 1
k = 100
c = 0.5

ccr = 2*np.sqrt(k*m)
seta = c/ccr
omegao = np.sqrt(k/m)
omegaamor = omegao*np.sqrt(1-seta**2)

omegamax = 40
omegaserie = np.linspace(0,omegamax,100,endpoint=True)
Fs = 2*np.max(omegaserie)/(2*np.pi)
dt = 1/Fs
dw = omegaserie[1]-omegaserie[0]
T = 2*np.pi/dw
N = int(2*omegamax/dw)
t = np.linspace(0,T,N,endpoint=True)
A1 = 1/(2*m*omegaamor*1j)
A2 = np.conj(A1)
lambda1 = -seta*omegao+1j*omegaamor

H = A1/(1j*omegaserie-lambda1)+A2/(1j*omegaserie-np.conj(lambda1))
h = 2*np.abs(A1)*np.exp(-seta*omegaamor*t)*np.sin(omegaamor*t)


plt.figure(1)
plt.subplot(1,2,1)
plt.plot(omegaserie,np.abs(H),'k')
plt.yscale('log')
#plt.xlim((-10,10))
plt.ylabel('$Abs(H(\omega))$')
plt.xlabel('$\omega$')
plt.subplot(1,2,2)
plt.plot(t,h,'k')
#plt.xlim((-10,10))
plt.ylabel('$h(t)$')
plt.xlabel('$t$')
plt.tight_layout()
plt.savefig('FRF_IRF.jpg',dpi=300)

