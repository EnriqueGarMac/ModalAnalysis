# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 19:27:28 2021

@author: Enrique GM
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from numpy.linalg import eig
import scipy.sparse.linalg as sla
from scipy.signal import find_peaks
import pandas as pd
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def MAC(phi_X, phi_A):
    """Modal Assurance Criterion.
    Literature:
        [1] Maia, N. M. M., and J. M. M. Silva. 
            "Modal analysis identification techniques." Philosophical
            Transactions of the Royal Society of London. Series A: 
            Mathematical, Physical and Engineering Sciences 359.1778 
            (2001): 29-40. 
    :param phi_X: Mode shape matrix X, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :param phi_A: Mode shape matrix A, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :return: MAC matrix. Returns MAC value if both ``phi_X`` and ``phi_A`` are
        one-dimensional arrays.
    """
    if phi_X.ndim == 1:
        phi_X = phi_X[:, np.newaxis]
    
    if phi_A.ndim == 1:
        phi_A = phi_A[:, np.newaxis]
    
    if phi_X.ndim > 2 or phi_A.ndim > 2:
        raise Exception(f'Mode shape matrices must have 1 or 2 dimensions (phi_X: {phi_X.ndim}, phi_A: {phi_A.ndim})')

    if phi_X.shape[0] != phi_A.shape[0]:
        raise Exception(f'Mode shapes must have the same first dimension (phi_X: {phi_X.shape[0]}, phi_A: {phi_A.shape[0]})')

    MAC = np.abs(np.conj(phi_X).T @ phi_A)**2
    for i in range(phi_X.shape[1]):
        for j in range(phi_A.shape[1]):
            MAC[i, j] = MAC[i, j]/\
                            (np.conj(phi_X[:, i]) @ phi_X[:, i] *\
                            np.conj(phi_A[:, j]) @ phi_A[:, j])

    
    if MAC.shape == (1, 1):
        MAC = MAC[0, 0]

    return MAC

def complex_to_normal_mode(mode, max_dof=50, long=True):
    """Transform a complex mode shape to normal mode shape.
    
    The real mode shape should have the maximum correlation with
    the original complex mode shape. The vector that is most correlated
    with the complex mode, is the real part of the complex mode when it is
    rotated so that the norm of its real part is maximized. [1]
    ``max_dof`` and ``long`` arguments are given for modes that have
    a large number of degrees of freedom. See ``_large_normal_mode_approx()``
    for more details.
    
    Literature:
        [1] Gladwell, H. Ahmadian GML, and F. Ismail. 
            "Extracting Real Modes from Complex Measured Modes."
    
    :param mode: np.ndarray, a mode shape to be transformed. Can contain a single
        mode shape or a modal matrix `(n_locations, n_modes)`.
    :param max_dof: int, maximum number of degrees of freedom that can be in
        a mode shape. If larger, ``_large_normal_mode_approx()`` function
        is called. Defaults to 50.
    :param long: bool, If True, the start in stepping itartion is altered, the
        angles of rotation are averaged (more in ``_large_normal_mode_approx()``).
        This is needed only when ``max_dof`` is exceeded. The normal modes are 
        more closely related to the ones computed with an entire matrix. Defaults to True.
    :return: normal mode shape
    """
    if mode.ndim == 1:
        mode = mode[None, :, None]
    elif mode.ndim == 2:
        mode = mode.T[:, :, None]
    else:
        raise Exception(f'`mode` must have 1 or 2 dimensions ({mode.ndim}).')
    
    if mode.shape[1] > max_dof:
        return _large_normal_mode_approx(mode[:, :, 0].T, step=int(np.ceil(mode.shape[1] / max_dof)) + 1, long=long)
    
    # Normalize modes so that norm == 1.0
    _norm = np.linalg.norm(mode, axis=1)[:, None, :]
    mode = mode / _norm

    mode_T = np.transpose(mode, [0, 2, 1])

    U = np.matmul(np.real(mode), np.real(mode_T)) + np.matmul(np.imag(mode), np.imag(mode_T))

    val, vec = np.linalg.eig(U)
    i = np.argmax(np.real(val), axis=1)

    normal_mode = np.real([v[:, _] for v, _ in zip(vec, i)]).T
    return normal_mode

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



def convert_mode(MOD, lh, lv):
    xun = np.array([0,0,0,0,0,lh,lh,lh,lh,lh,np.nan,0,lh,np.nan,0,lh,np.nan,0,lh,np.nan,0,lh])
    yun = np.array([0,lv,2*lv,3*lv,4*lv,4*lv,3*lv,2*lv,lv,0,np.nan,lv,lv,np.nan,2*lv,2*lv,np.nan,3*lv,3*lv,np.nan,4*lv,4*lv])
    xde = xun.copy()
    yde = yun.copy()
    xde[1] = xde[1]+MOD[0]
    xde[2] = xde[2]+MOD[1]
    xde[3] = xde[3]+MOD[2]
    xde[4] = xde[4]+MOD[3]
    xde[5] = xde[5]+MOD[3]
    xde[6] = xde[6]+MOD[2]
    xde[7] = xde[7]+MOD[1]
    xde[8] = xde[8]+MOD[0]
    xde[11] = xde[11]+MOD[0]
    xde[12] = xde[12]+MOD[0]
    xde[14] = xde[14]+MOD[1]
    xde[15] = xde[15]+MOD[1]
    xde[17] = xde[17]+MOD[2]
    xde[18] = xde[18]+MOD[2]
    xde[20] = xde[20]+MOD[3]
    xde[21] = xde[21]+MOD[3]
    return xde,yde

def plotframemodes(mult, MOD, TMS, lh, lv, TFreq, Expfreq):
    plt.clf()
    xun,yun = convert_mode(MOD[:,0]*0, lh, lv)
    MOD = MOD*mult
    plt.subplot(1,4,1)
    plt.axis('equal')
    plt.xlim([-4,10])
    sign = np.matmul(MOD[:,0].T,TMS[:,0])/(np.matmul(TMS[:,0].T,TMS[:,0]))
    xde,yde = convert_mode(MOD[:,0], lh, lv)
    xdet,ydet = convert_mode(sign*TMS[:,0], lh, lv)
    plt.plot(xde,yun,'-bo',linewidth=3)
    plt.plot(xdet,ydet,'--r')
    plt.plot(xun,yun,'--k')
    plt.title('Modo 1 - Freq. teorica/exp :'+str(round(float(TFreq[0]), 2))+'/'+str(round(float(Expfreq[0]), 2)))
    plt.subplot(1,4,2)
    plt.axis('equal')
    plt.xlim([-4,10])
    sign = np.matmul(MOD[:,1].T,TMS[:,1])/(np.matmul(TMS[:,1].T,TMS[:,1]))
    xde,yde = convert_mode(MOD[:,1], lh, lv)
    xdet,ydet = convert_mode(sign*TMS[:,1], lh, lv)
    plt.plot(xde,yun,'-bo',linewidth=3)
    plt.plot(xdet,ydet,'--r')
    plt.plot(xun,yun,'--k')
    plt.title('Modo 2 - Freq. teorica/exp :'+str(round(float(TFreq[1]), 2))+'/'+str(round(float(Expfreq[1]), 2)))
    plt.subplot(1,4,3)
    plt.axis('equal')
    plt.xlim([-4,10])
    sign = np.matmul(MOD[:,2].T,TMS[:,2])/(np.matmul(TMS[:,2].T,TMS[:,2]))
    xde,yde = convert_mode(MOD[:,2], lh, lv)
    xdet,ydet = convert_mode(sign*TMS[:,2], lh, lv)
    plt.plot(xde,yun,'-bo',linewidth=3)
    plt.plot(xdet,ydet,'--r')
    plt.plot(xun,yun,'--k')
    plt.title('Modo 3 - Freq. teorica/exp :'+str(round(float(TFreq[2]), 2))+'/'+str(round(float(Expfreq[2]), 2)))
    plt.subplot(1,4,4)
    plt.axis('equal')
    plt.xlim([-4,10])
    sign = np.matmul(MOD[:,3].T,TMS[:,3])/(np.matmul(TMS[:,3].T,TMS[:,3]))
    xde,yde = convert_mode(MOD[:,3], lh, lv)
    xdet,ydet = convert_mode(sign*TMS[:,3], lh, lv)
    plt.plot(xde,yun,'-bo',linewidth=3)
    plt.plot(xdet,ydet,'--r')
    plt.plot(xun,yun,'--k')
    plt.title('Modo 4 - Freq. teorica/exp :'+str(round(float(TFreq[3]), 2))+'/'+str(round(float(Expfreq[3]), 2)))
    plt.tight_layout()
    plt.show()
