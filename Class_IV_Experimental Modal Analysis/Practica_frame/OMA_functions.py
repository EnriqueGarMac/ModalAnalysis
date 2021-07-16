# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:15:29 2021

@author: Enrique GM
"""

import scipy.sparse.linalg as sla
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.signal import welch
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import welch
from scipy.linalg import expm



def Rayleighdamping(seta1, seta2, w1, w2):
    B=np.array([[1/(2*w1),w1/2],[1/(2*w2),w2/2]])
    A=np.matmul(np.linalg.inv(B),np.array([[seta1],[seta2]]))
    alpha0=A[0]
    alpha1=A[1]
    return alpha0,alpha1


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
  Mred = Mred.astype('float')
  Kred = Kred.astype('float')
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

def deriv(state, t, A, B, forcevector,tf):
    forceint = np.zeros((np.size(forcevector,1),1))
    for ij in np.arange(0,np.size(forcevector,1)):
        f = interp1d(tf, forcevector[:,ij],kind='linear', fill_value="extrapolate")
        forceint[ij,0]= f(t)
    dydx = np.dot(A,state)+np.dot(B,forceint)[:,0]
    return dydx

def statespace(K, C, M, B2):
    ndofs = np.shape(K)[0]
    A = np.zeros([2*ndofs, 2*ndofs])
    A[0:ndofs, ndofs:2*ndofs] = np.eye(ndofs)
    A[ndofs:2*ndofs, 0:ndofs] = -np.dot(np.linalg.inv(M), K)
    A[ndofs:2*ndofs, ndofs:2*ndofs] = -np.dot(np.linalg.inv(M), C)
    
    B = np.zeros((2*ndofs,1))
    B[ndofs:2*ndofs, :] = np.dot(np.linalg.inv(M),B2)
    return A,B

def statespace_discrete(K, C, M, B2, dt):
    
    # Continuous
    ndofs = np.shape(K)[0]
    Ac = np.zeros([2*ndofs, 2*ndofs])
    Ac[0:ndofs, ndofs:2*ndofs] = np.eye(ndofs)
    Ac[ndofs:2*ndofs, 0:ndofs] = -np.dot(np.linalg.inv(M), K)
    Ac[ndofs:2*ndofs, ndofs:2*ndofs] = -np.dot(np.linalg.inv(M), C)
    
    Bc = np.zeros((2*ndofs,1))
    Bc[ndofs:2*ndofs, :] = np.dot(np.linalg.inv(M),B2)
    
    # Discrete
    A = expm(Ac*dt)
    B = np.dot(np.dot((A-np.eye(np.size(A,0))),np.linalg.inv(Ac)),Bc)
      
    return A,B

def state_space_solver(A,B,forcevector,iforce,uo,vo,dt):
      # Solve the state-space equation
      A0 = np.concatenate((uo,vo),axis=0)
      N = np.size(forcevector,0)
      tf=np.linspace(0,(N-1)*dt,N, endpoint=True)
      y = integrate.odeint(deriv, A0[:,0], tf, args=(A,B,forcevector,tf,iforce),rtol = 1e-6, atol=1e-18)
      q = y[:,0:np.size(K,1)]       # Displacement
      v =y[:,np.size(K,1):]   # Velocity
      q = np.delete(q, np.size(q,1), 0)
      a=q*0
      for i in np.arange(0,np.size(K,1)):
         a[:,i] = np.diff(v[:,i])/dt              # Acceleration
      v = np.delete(v, np.size(v,1), 0)
      tf = np.delete(tf, np.size(tf,0)-1, 0)
      return [q,v,a,tf];

def state_space_solver_comp(K,C,M,forcevector,B2,uo,vo,dt):
    
      A,B = statespace(K, C, M, B2)
    
      # Solve the state-space equation
      if np.size(uo,0)<np.size(uo,1):
          uo = uo.T
      if np.size(vo,0)<np.size(vo,1):
          vo = vo.T
      A0 = np.concatenate((uo,vo),axis=0)
      N = np.size(forcevector,0)
      tf=np.linspace(0,(N-1)*dt,N, endpoint=True)
      y = integrate.odeint(deriv, A0[:,0], tf, args=(A,B,forcevector,tf),rtol = 1e-6, atol=1e-18)
      q = y[:,0:np.size(K,1)]       # Displacement
      v =y[:,np.size(K,1):]   # Velocity
      q = np.delete(q, np.size(q,1), 0)
      a=q*0
      for i in np.arange(0,np.size(K,1)):
         a[:,i] = np.diff(v[:,i])/dt              # Acceleration
      v = np.delete(v, np.size(v,1), 0)
      tf = np.delete(tf, np.size(tf,0)-1, 0)
      return [q,v,a,tf];
  
def state_space_solver_comp_dis(K,C,M,forcevector,B2,uo,vo,dt):
    
      A,B = statespace_discrete(K, C, M, B2, dt)

      # Solve the discrete state-space equation
      if np.size(uo,0)<np.size(uo,1):
          uo = uo.T
      if np.size(vo,0)<np.size(vo,1):
          vo = vo.T
      A0 = np.concatenate((uo,vo),axis=0)
      
      N = int(np.size(forcevector,0))
      forcevector.reshape((N,np.size(forcevector,1)))
      
      y = np.zeros((N,np.size(K,1)*2))
      tf = np.zeros((N,1))
      y[0,:] = A0.T
      tf[0] = 0.
      for j in np.arange(1,N,1):
          y[j,:] = np.dot(A,y[j-1,:])+np.dot(B,forcevector[j,0:])
          tf[j] = j*dt
      
      q = y[:,0:np.size(K,1)]       # Displacement
      v =y[:,np.size(K,1):]   # Velocity
      q = np.delete(q, np.size(q,1), 0)
      a=q*0
      for i in np.arange(0,np.size(K,1)):
         a[:,i] = np.diff(v[:,i])/dt              # Acceleration
      v = np.delete(v, np.size(v,1), 0)
      tf = np.delete(tf, np.size(tf,0)-1, 0)
      return [q,v,a,tf];