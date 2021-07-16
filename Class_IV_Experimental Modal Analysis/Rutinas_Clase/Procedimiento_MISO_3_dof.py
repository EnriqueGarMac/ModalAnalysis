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


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]

def eigen2(K,M,b,nmodes):
  # [L,X]=eigen(K,M,b)
  #-------------------------------------------------------------
  # PURPOSE
  #  Solve the generalized eigenvalue problem
  #  [K-LM]X = 0, considering boundary conditions.
  #
  # INPUT:
  #    K : global stiffness matrix, dim(K)= nd x nd
  #    M : global mass matrix, dim(M)= nd x nd
  #    b : boundary condition matrix
  #        dim(b)= nb x 1
  # OUTPUT:
  #    L : eigenvalues stored in a vector with length (nd-nb) 
  #    X : eigenvectors dim(X)= nd x nfdof, nfdof : number of dof's
  #-------------------------------------------------------------
  #-------------------------------------------------------------
  [nd,nd]=np.shape(K);
  fdof=np.array([np.arange(1,nd+1,1)]).T

  pdof = b.astype(int);
  fdof = np.delete(fdof, pdof)
  fdof = fdof.astype(int)
  Kred = np.delete(K, pdof, axis=1)
  Kred = np.delete(Kred, pdof, axis=0)
  Mred = np.delete(M, pdof, axis=1)
  Mred = np.delete(Mred, pdof, axis=0)
  [D,X1]=sla.eigs(Kred,nmodes,Mred, which='SM')
  D = np.real(D);
  X1 = np.real(X1);
  [nfdof,nfdof]=np.shape(X1);
  for j in np.arange(0,nfdof,1):
        mnorm=np.sqrt(np.dot(np.dot(X1[:,j].T,Mred),(X1[:,j])));
        X1[:,j]=X1[:,j]/mnorm;
  i=np.argsort(D)
  L=np.sort(D);
  X2=X1[:,i];
  X=np.zeros((nd,nfdof))
  for n in np.arange(1,nmodes,1):
        X[fdof-1,n-1]=X2[:,n-1];

  
  return [L,X];


wserie = np.linspace(0,500,50000, endpoint = True)

k1 = 7.5*10**7
k2 = k1
k3 = k1
k4 = k1
m1 = 3600
m2 = 2850
m3 = m2
m4 = 1800

kelem = [k1,k2,k3,k4]
M = np.array([[m1,0,0,0],[0,m2,0,0],[0,0,m3,0],[0,0,0,m4]])
K = np.array([[kelem[0]+kelem[1],-kelem[1],0,0],
              [-kelem[1],kelem[1]+kelem[2],-kelem[2],0],
              [0,-kelem[2],kelem[2]+kelem[3],-kelem[3]],
              [0,0,-kelem[3],kelem[3]]])


# MODAL ANALYSIS
[L,MOD]=eigen2(K,M,np.array([]),5);
wi = np.sqrt(L)
MOD = np.dot(MOD,np.diag(np.sqrt((1/np.diag(np.dot(MOD.T,(np.dot(M,MOD))))))))
                            
seta1 = 0.01
seta2 = 0.01
seta3 = 0.01
seta4 = 0.01
H1 = 1/(wi[0]**2-wserie**2+1j*2*seta1*wi[0]*wserie)
H2 = 1/(wi[1]**2-wserie**2+1j*2*seta2*wi[1]*wserie)
H3 = 1/(wi[2]**2-wserie**2+1j*2*seta3*wi[2]*wserie)
H4 = 1/(wi[3]**2-wserie**2+1j*2*seta4*wi[3]*wserie)

Hm = np.vstack((np.vstack((np.vstack((H1,H2)),H3)),H4))
l = 3
k = 3
Hlk = np.dot(MOD[l,:]*MOD[k,:],Hm)
MOD[k,:].T
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(wserie,np.abs(H1),'r', label = '$H_1$')
plt.plot(wserie,np.abs(H2),'g', label = '$H_2$')
plt.plot(wserie,np.abs(H3),'b', label = '$H_3$')
plt.plot(wserie,np.abs(H4),'y', label = '$H_4$')
plt.legend()
plt.xlim((0,500))
plt.ylabel('$Abs(H(\omega))$')
plt.xlabel('$\omega$')
plt.yscale('log')
plt.subplot(2,1,2)
plt.plot(wserie,np.abs(Hlk),'k')
plt.xlim((0,500))
plt.ylabel('$Abs(H_{lk}(\omega))$')
plt.xlabel('$\omega$')
plt.yscale('log')
plt.tight_layout()
plt.savefig('3DOF_modal_decomposition.jpg',dpi=300)


plt.figure(2)
peaks, _ = find_peaks(np.abs(Hlk), height=0)
plt.plot(wserie,np.abs(Hlk),'k')
plt.plot(wserie[peaks], np.abs(Hlk)[peaks], "x")
plt.stem(wserie[peaks],np.abs(Hlk)[peaks], '--k')
plt.show()
plt.xlim((0,500))
plt.ylim((0.1*np.min(np.abs(Hlk)),10*np.max(np.abs(Hlk))))
plt.ylabel('$Abs(H_{lk}(\omega))$')
plt.xlabel('$\omega$')
plt.yscale('log')
plt.tight_layout()
plt.savefig('3DOF_peak_picking.jpg',dpi=300)

freqpeak = wserie[peaks]
amppeak = np.abs(Hlk)[peaks]


print(freqpeak)
print(wi)


plt.figure(3)
peaks, _ = find_peaks(np.abs(Hlk), height=0)
plt.plot(wserie,np.abs(Hlk),'k')
plt.plot(wserie[peaks], np.abs(Hlk)[peaks], "x")
plt.stem(wserie[peaks],np.abs(Hlk)[peaks], '--k')
damp = np.zeros((len(freqpeak),1))
for j in np.arange(0,len(freqpeak),1):
    valsearch = amppeak[j]/np.sqrt(2)
    searchband = np.arange(peaks[j]-500,peaks[j])
    ampband = np.abs(Hlk)[searchband]
    freqband = wserie[searchband]
    idx,val1 = find_nearest(ampband,valsearch)
    f1 = freqband[idx]
    searchband = np.arange(peaks[j],peaks[j]+500)
    ampband = np.abs(Hlk)[searchband]
    freqband = wserie[searchband]
    idx,val2 = find_nearest(ampband,valsearch)
    f2 = freqband[idx]
    damp[j] = (f2-f1)/(f2+f1)
    plt.plot(np.array([f1,f2]),np.array([val1,val2]),'r')

plt.show()
plt.xlim((0,500))
plt.ylim((0.1*np.min(np.abs(Hlk)),10*np.max(np.abs(Hlk))))
plt.ylabel('$Abs(H_{lk}(\omega))$')
plt.xlabel('$\omega$')
plt.yscale('log')
plt.tight_layout()
plt.savefig('3DOF_peak_picking_damping.jpg',dpi=300)
print(damp)


seta1exp = damp[0]
seta2exp = damp[1]
seta3exp = damp[2]
seta4exp = damp[3]
wiexp = freqpeak

Expfreq = freqpeak/(2*np.pi)
print('Damping')
print(damp.T)
print('Resonant frequencies')
print(Expfreq)