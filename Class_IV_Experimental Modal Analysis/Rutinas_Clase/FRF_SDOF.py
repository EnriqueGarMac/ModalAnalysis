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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def FRFana(ReX, ReY, seta, wo):

    N = np.size(ReX,0)
    ReH = np.zeros((N,N)).astype(dtype=float)
    ImagH = np.zeros((N,N)).astype(dtype=float)
    PhH = np.zeros((N,N)).astype(dtype=float)
    AbsH = np.zeros((N,N)).astype(dtype=float)
    for i in range(0, N):
        for j in range(0, N):
         p =  ReX[i,j]+1j*ReY[i,j]
         FRF = 1/(wo**2+p**2+2*seta*wo*p)
         ReH[i,j] = np.real(FRF)
         ImagH[i,j] = np.imag(FRF)
         PhH[i,j] = np.angle(FRF)
         AbsH[i,j] = np.abs(FRF)
         if AbsH[i,j]>1:
             AbsH[i,j]=1
         if ImagH[i,j]>1:
             ImagH[i,j]=1
         if ReH[i,j]>1:
             ReH[i,j]=1
         if AbsH[i,j]<-1:
             AbsH[i,j]=-1
         if ImagH[i,j]<-1:
             ImagH[i,j]=-1
         if ReH[i,j]<-1:
             ReH[i,j]=-1

    return ReH, ImagH, PhH, AbsH



wserie = np.linspace(-10,10,200, endpoint = True)
wo = 5
seta = 0.05
FRF = 1/(wo**2-wserie**2+1j*2*seta*wo*wserie)


plt.figure(1)
plt.subplot(2,2,1)
plt.plot(wserie,np.real(FRF),'k')
plt.xlim((-10,10))
plt.ylabel('$Re(H(\omega))$')
plt.xlabel('$\omega$')
plt.subplot(2,2,2)
plt.plot(wserie,np.imag(FRF),'k')
plt.xlim((-10,10))
plt.ylabel('$Imag(H(\omega))$')
plt.xlabel('$\omega$')
plt.subplot(2,2,3)
plt.plot(wserie,np.abs(FRF),'k')
plt.xlim((-10,10))
plt.ylabel('$Abs(H(\omega))$')
plt.xlabel('$\omega$')
plt.subplot(2,2,4)
plt.plot(wserie,np.angle(FRF),'k')
plt.xlim((-10,10))
plt.ylabel('$Phase(H(\omega))$')
plt.xlabel('$\omega$')
plt.tight_layout()
plt.savefig('Transfer_f_SDOF.jpg',dpi=300)


fig = plt.figure(2)
ax = fig.gca(projection='3d')

pserie2 = np.linspace(-40,40,100, endpoint = True)
pserie1 = np.linspace(-1,0,100, endpoint = True)
ReX, ReY = np.meshgrid(pserie1, pserie2)

ReH, ImagH, PhH, AbsH = FRFana(ReX, ReY, seta, wo)
Z = ReH

# Plot the surface.
surf = ax.plot_surface(ReX, ReY, Z,rstride=1, cstride=1, cmap='viridis',
                       linewidth=0.1, antialiased=False, edgecolor='none',alpha=0.4)
ax.plot(ReX[:,0],ReY[:,0]+0.1,Z[:,0],'k',linewidth=0)
ax.plot(ReX[:,20],ReY[:,0]+0.1,Z[:,20],'k',linewidth=2)
ax.plot(ReX[:,40],ReY[:,0]+0.1,Z[:,40],'k',linewidth=2)
ax.plot(ReX[:,60],ReY[:,0]+0.1,Z[:,60],'k',linewidth=2)
ax.plot(ReX[:,80],ReY[:,0]+0.1,Z[:,80],'k',linewidth=2)
ax.plot(ReX[:,99],ReY[:,0]+0.1,Z[:,99],'k',linewidth=4)

# Customize the z axis.
ax.set_zlim(-1, 1)
ax.set_xlabel('$Re(H(p))$')
ax.set_ylabel('$Imag(H(p))$')
ax.set_zlabel('$Re(H(p))$')
ax.view_init(28,-36)
plt.show()
plt.savefig('Re_3D.jpg',dpi=300)

fig = plt.figure(3)
ax = fig.gca(projection='3d')

Z = ImagH

# Plot the surface.
surf = ax.plot_surface(ReX, ReY, Z,rstride=1, cstride=1, cmap='viridis',
                       linewidth=0.1, antialiased=False, edgecolor='none',alpha=0.4)
ax.plot(ReX[:,0],ReY[:,0]+0.1,Z[:,0],'k',linewidth=0)
ax.plot(ReX[:,20],ReY[:,0]+0.1,Z[:,20],'k',linewidth=2)
ax.plot(ReX[:,40],ReY[:,0]+0.1,Z[:,40],'k',linewidth=2)
ax.plot(ReX[:,60],ReY[:,0]+0.1,Z[:,60],'k',linewidth=2)
ax.plot(ReX[:,80],ReY[:,0]+0.1,Z[:,80],'k',linewidth=2)
ax.plot(ReX[:,99],ReY[:,0]+0.1,Z[:,99],'k',linewidth=4)

# Customize the z axis.
ax.set_zlim(-1, 1)
ax.set_xlabel('$Re(H(p))$')
ax.set_ylabel('$Imag(H(p))$')
ax.set_zlabel('$Imag(H(p))$')
ax.view_init(28,-36)
plt.show()
plt.savefig('Imag_3D.jpg',dpi=300)


fig = plt.figure(4)
ax = fig.gca(projection='3d')

Z = AbsH

# Plot the surface.
surf = ax.plot_surface(ReX, ReY, Z,rstride=1, cstride=1, cmap='viridis',
                       linewidth=0.1, antialiased=False, edgecolor='none',alpha=0.4)
ax.plot(ReX[:,0],ReY[:,0]+0.1,Z[:,0],'k',linewidth=0)
ax.plot(ReX[:,20],ReY[:,0]+0.1,Z[:,20],'k',linewidth=2)
ax.plot(ReX[:,40],ReY[:,0]+0.1,Z[:,40],'k',linewidth=2)
ax.plot(ReX[:,60],ReY[:,0]+0.1,Z[:,60],'k',linewidth=2)
ax.plot(ReX[:,80],ReY[:,0]+0.1,Z[:,80],'k',linewidth=2)
ax.plot(ReX[:,99],ReY[:,0]+0.1,Z[:,99],'k',linewidth=4)

# Customize the z axis.
ax.set_zlim(-1, 1)
ax.set_xlabel('$Re(H(p))$')
ax.set_ylabel('$Imag(H(p))$')
ax.set_zlabel('$Abs(H(p))$')
ax.view_init(28,-36)
plt.show()
plt.savefig('Abs_3D.jpg',dpi=300)


fig = plt.figure(5)
ax = fig.gca(projection='3d')

Z = PhH

# Plot the surface.
surf = ax.plot_surface(ReX, ReY, Z,rstride=1, cstride=1, cmap='viridis',
                       linewidth=0.1, antialiased=False, edgecolor='none',alpha=0.4)
ax.plot(ReX[:,0],ReY[:,0]+0.1,Z[:,0],'k',linewidth=0)
ax.plot(ReX[:,20],ReY[:,0]+0.1,Z[:,20],'k',linewidth=2)
ax.plot(ReX[:,40],ReY[:,0]+0.1,Z[:,40],'k',linewidth=2)
ax.plot(ReX[:,60],ReY[:,0]+0.1,Z[:,60],'k',linewidth=2)
ax.plot(ReX[:,80],ReY[:,0]+0.1,Z[:,80],'k',linewidth=2)
ax.plot(ReX[:,99],ReY[:,0]+0.1,Z[:,99],'k',linewidth=4)

# Customize the z axis.
ax.set_xlabel('$Re(H(p))$')
ax.set_ylabel('$Imag(H(p))$')
ax.set_zlabel('$Phase(H(p))$')
ax.view_init(28,-36)
plt.show()
plt.savefig('Phase_3D.jpg',dpi=300)

