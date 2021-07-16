# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 23:58:53 2021

@author: Enrique GM
"""

import numpy as np
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate as interp
from matplotlib.colors import LightSource
#import FEM_functions as f
from FEM_functions import *

# ------------------------------------------------
# MODAL ANALYSIS SHELLS WITH FIRST ORDER DEFORMATION THEORY
# ------------------------------------------------
# Author: E. García-Macías

# GEOMETRY
a=10/100               # m 
b=10/100               # m
h=a*0.1             # Plate's thickness (m)

# MATERIAL
Em=2.1E+11       # Youngs modulus
poism=0.3       # Poissons ratio
dens=7850
ks=5/6
P=-1            # Uniform transverse pressure

# MESHING PARAMETERS:
ndivx=12           # Number of divisions in x direction
ndivy=12           # Number of divisions in y direction

# FEM
# *************************************************************************

x=np.arange(0,a/ndivx,a)
y=np.arange(0,b/ndivy,b)

[Coordelemx,Coordelemy]=ElemCoordxy(ndivx,ndivy,a,b)
Coordelemz=Coordelemx*0
[Edof, nodos]=ElemTipology(ndivx,ndivy,5)

K=np.zeros((5*(ndivx+1)*(ndivy+1),5*(ndivx+1)*(ndivy+1)))
M=np.zeros((5*(ndivx+1)*(ndivy+1),5*(ndivx+1)*(ndivy+1)))
f=np.zeros((1,np.shape(K)[0]))


for n in np.arange(1,(ndivx*ndivy)+1,1):
    ex=Coordelemx[n-1,:]
    ey=Coordelemy[n-1,:]
    ez=Coordelemz[n-1,:]
    ep=np.array([Em,poism,dens,ks])
    [Ke,Me,fe]=shell_midlin(ex,ey,ez,ep,h,P)
    [K,M,f]=assem(Edof[n-1,1:],K,Ke,M,Me,f,fe)

# BOUNDARY CONDITIONS
bc=SSSS(ndivx,ndivy,5)


# MODAL ANALYSIS
[L,MOD]=eigen2(K,M,bc,100);
frequenciesHz=np.sqrt(L)/(2*np.pi);

modopolot = 4


modetoplot=MOD[:,modopolot]
U1 = modetoplot[0:-1:5]
U2 = modetoplot[1:-1:5]
U3 = modetoplot[2:-1:5]

despelemx=np.zeros((ndivx+1,ndivy+1))
despelemy=np.zeros((ndivx+1,ndivy+1))
despelemz=np.zeros((ndivx+1,ndivy+1))

nod=0;
for i in np.arange(0,ndivy,1):
    for j in np.arange(0,ndivx,1):
      despelemx[j,i]=U1[nod]
      despelemy[j,i]=U2[nod]
      despelemz[j,i]=U3[nod]
      nod=nod+1;
    nod=nod+1;

X = np.linspace(0,a,ndivx+1)
Y = np.linspace(0,b,ndivy+1)
X, Y = np.meshgrid(X, Y)

x_dense = np.linspace(0,a,5*(ndivx+1))
y_dense = np.linspace(0,b,5*(ndivy+1))
x_dense, y_dense = np.meshgrid(x_dense, y_dense)

zfun_smooth_rbf = interp.Rbf(X, Y, despelemz, function='linear', smooth=0.0)  # default smooth=0 for interpolation
z_dense_smooth_rbf = zfun_smooth_rbf(x_dense, y_dense)  # not really a function, but a callable class instance


fig = plt.figure()
ax = fig.gca(projection='3d')
# Create light source object.
ls = LightSource(azdeg=80, altdeg=20)
# Shade data, creating an rgb array.
rgb = ls.shade(z_dense_smooth_rbf, plt.cm.viridis)
surf = ax.plot_surface(x_dense, y_dense, z_dense_smooth_rbf, rstride=1, cstride=1, linewidth=0,
                       antialiased=False, facecolors=rgb)

# Add contour lines to further highlight different levels.
ax.contour(x_dense, y_dense, z_dense_smooth_rbf, levels=[-3,0,3], linestyles='-', offset=np.min(z_dense_smooth_rbf))
plt.title('Modo '+ str(modopolot) +' Frequency '+str(round(frequenciesHz[modopolot],2))+ 'Hz')
plt.show()
