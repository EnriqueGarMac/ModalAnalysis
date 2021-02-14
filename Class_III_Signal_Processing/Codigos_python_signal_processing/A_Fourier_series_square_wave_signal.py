# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 19:44:44 2021

@author: Enrique GM
Example - Fourier series of square wave signal
"""


import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0,1,800)
Nserie = np.array([2,4,10,20,40,100])

plt.figure(1)
plt.plot([0,0,1/2,1/2,1],[0,1,1,0,0],'-k',linewidth=2, label="Original signal")

for j in np.arange(0,len(Nserie),1):
  N = Nserie[j]
  f = np.ones([1,800])*1/2;
  #leyenda.append('n='+N)
  for i  in np.arange(1,N,2):
    a = 2.0/np.pi/i
    f = f+ a*np.sin(2*np.pi*i*x);
  plt.plot(x,f.T,linewidth=1, label='n='+str(N))

plt.legend()
plt.xlabel('t')
plt.ylabel('x(t)')
plt.ylim([-0.2,1.2])
plt.show()
plt.savefig('Fourier_square.jpg',dpi=300)