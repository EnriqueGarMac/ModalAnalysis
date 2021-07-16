# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 14:32:03 2021

@author: Enrique GM
"""

import numpy as np
import matplotlib.pyplot as plt


# Single Degree Of Freedom (SDOF) dynamic system:


k = 10;        # Stiffness
m = 1;         # Mass
seta = 5/100;  # Damping ratio
uo = 0.1;        # Initial displacement
vo = 0;        # Initial velocity

omega = np.sqrt(k/m);
omegad = omega*np.sqrt(1-seta**2);

t = np.arange(0,30,0.1)
xfree = np.zeros((1,np.size(t)))
# BCs
An = uo;
Bn = (vo+seta*omega*An)/(omegad);

for i in np.arange(0,np.size(t),1):
   xfree[0,i] = np.exp(-seta*omega*t[i])*(An*np.cos(omegad*t[i])+Bn*np.sin(omegad*t[i])); 


plt.figure(1,figsize=(7,7))
plt.plot(t,xfree[0,:],linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Displacement [m]')
plt.savefig('SDOF_disp_fv.jpg',dpi=300)


# Dynamic amplification factor
setaserie = np.array([0.1,1.0,2.0,4.0,10.0])/100;


beta = np.linspace(0,3,400,endpoint=True);
D = np.zeros((np.size(beta),np.size(setaserie)))
for ij in np.arange(0,np.size(setaserie)):
    for iijj in np.arange(0,np.size(beta)):
      D[iijj,ij] = ((1-beta[iijj]**2)**2+(2*setaserie[ij]*beta[iijj])**2)**(-1/2); 

plt.figure(2)
plt.plot(beta,D[:,0],linewidth=2,label = r"$\zeta=0.1\%$")
plt.plot(beta,D[:,1],linewidth=2,label = r"$\zeta=1\%$")
plt.plot(beta,D[:,2],linewidth=2,label = r"$\zeta=2\%$")
plt.plot(beta,D[:,3],linewidth=2,label = r"$\zeta=4\%$")
plt.plot(beta,D[:,4],linewidth=2,label = r"$\zeta=10\%$")
plt.ylabel('D')
plt.xlabel(r"$\beta$")
plt.legend()
plt.yscale('log')
plt.savefig('Amplificacion_factor.jpg',dpi=300)