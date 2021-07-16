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
    
    B = np.zeros((2*ndofs,ndofs))
    B[ndofs:2*ndofs, :] = np.dot(np.linalg.inv(M),B2)
    return A,B

def statespace_discrete(K, C, M, B2, dt):
    
    # Continuous
    ndofs = np.shape(K)[0]
    Ac = np.zeros([2*ndofs, 2*ndofs])
    Ac[0:ndofs, ndofs:2*ndofs] = np.eye(ndofs)
    Ac[ndofs:2*ndofs, 0:ndofs] = -np.dot(np.linalg.inv(M), K)
    Ac[ndofs:2*ndofs, ndofs:2*ndofs] = -np.dot(np.linalg.inv(M), C)
    
    Bc = np.zeros((2*ndofs,ndofs))
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

def plotframemodes(mult, MOD, lh, lv, Expfreq):
    plt.clf()
    xun,yun = convert_mode(MOD[:,0]*0, lh, lv)
    maxmod = np.max(MOD,1)
    maxmod
    MOD = mult*2*MOD/maxmod
    plt.subplot(1,4,1)
    plt.axis('equal')
    plt.xlim([-4,10])
    xde,yde = convert_mode(MOD[:,0], lh, lv)
    plt.plot(xde,yun,'-bo',linewidth=3)
    plt.plot(xun,yun,'--k')
    plt.title('Modo 1 - Freq. : '+str(round(float(Expfreq[0]), 2)))
    plt.subplot(1,4,2)
    plt.axis('equal')
    plt.xlim([-4,10])
    xde,yde = convert_mode(MOD[:,1], lh, lv)
    plt.plot(xde,yun,'-bo',linewidth=3)
    plt.plot(xun,yun,'--k')
    plt.title('Modo 2 - Freq. : '+str(round(float(Expfreq[1]), 2)))
    plt.subplot(1,4,3)
    plt.axis('equal')
    plt.xlim([-4,10])
    xde,yde = convert_mode(MOD[:,2], lh, lv)
    plt.plot(xde,yun,'-bo',linewidth=3)
    plt.plot(xun,yun,'--k')
    plt.title('Modo 3 - Freq. : '+str(round(float(Expfreq[2]), 2)))
    plt.subplot(1,4,4)
    plt.axis('equal')
    plt.xlim([-4,10])
    xde,yde = convert_mode(MOD[:,3], lh, lv)
    plt.plot(xde,yun,'-bo',linewidth=3)
    plt.plot(xun,yun,'--k')
    plt.title('Modo 4 - Freq. : '+str(round(float(Expfreq[3]), 2)))
    plt.tight_layout()
    plt.show()

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