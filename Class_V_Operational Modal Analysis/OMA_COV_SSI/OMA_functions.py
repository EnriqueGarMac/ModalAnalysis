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
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import welch
from scipy.linalg import expm
from scipy.signal import csd
from numpy.linalg import svd
import pandas as pd
import pip
from scipy.signal import correlate
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster

package = 'mplcursors'
def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package]) 
import mplcursors


def find_nearest(array, value):
     array = np.asarray(array)
     idx = (np.abs(array - value)).argmin()
     return int(idx)

def Rayleighdamping(seta1, seta2, w1, w2):
    B=np.array([[1/(2*w1),w1/2],[1/(2*w2),w2/2]])
    A=np.matmul(np.linalg.inv(B),np.array([[seta1],[seta2]]))
    alpha0=A[0]
    alpha1=A[1]
    return alpha0,alpha1

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

    MAC = np.real(np.abs(np.conj(phi_X).T @ phi_A)**2)
    for i in range(phi_X.shape[1]):
        for j in range(phi_A.shape[1]):
            MAC[i, j] = MAC[i, j]/\
                            (np.conj(phi_X[:, i]) @ phi_X[:, i] *\
                            np.conj(phi_A[:, j]) @ phi_A[:, j])

    
    if MAC.shape == (1, 1):
        MAC = MAC[0, 0]

    return MAC


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
  
    
  
# ***********************************************************************************************
# ***********************************************************************************************
# ***********************************************************************************************
#                                       FDD
# ***********************************************************************************************
# ***********************************************************************************************
# ***********************************************************************************************
# ***********************************************************************************************


    
def mainFDD(inputMatrix,sampleFreq,title,figTitle,FDDsolverTitle, peakThresh=-10000, frequencyThresh = 60, nfft = 2**11):
    # Frequency Domain Decomposition
    # Input:
    #     inputMatrix        -    [m,n] matrix where each column n contains m sample measurements from one channel
    #     sampleFreq         -    Number of samples per second in Hz
    #     peakThresh         -    Only the peaks with values above this threshold will be enumerated.
    #
    # Output: 
    #     Frequencies        -    Array of frequencies corresponding to the FDD-plot
    #     dbs1               -    Array of the absolute value of the identified singular values given in decibel.
    #     chosenPeaksFreq    -    Array of the frequencies corresponding to the peaks chosen by the user
    #     chosenPeaksMag     -    Array of the magnitude of the peaks chosen by the user
    #     chosenPeaksMS      -    Matrix [n,k] containing k modeshapes. Each row in n corresponds to the channel position n in inputMatrix.
    
    
    
    #Allocating space
   
    rows = np.size(inputMatrix,1)
    cols = np.size(inputMatrix,1)
    trial = csd(inputMatrix[:,0],inputMatrix[:,1],sampleFreq, nperseg=nfft)
    depth = len(trial[0])
    PSD_matrix = np.empty((rows,cols,depth),dtype='complex64')
    freq_matrix = np.empty((rows,cols,depth),dtype='float')
    Frequencies = np.empty(np.size(freq_matrix,2),dtype='float')
    Frequencies[:] = trial[0]
    
    
    #Compute PSD matrix, PSD_matrix[i,j,k]  where [i,j,:] contain the cross-spectra density of input channel i and j. 
    #    Each k corresponds to a frequency step derived from the given sample rate, so there is a 2D PSD matrix for each frequency step k.
    
    for i in range(np.size(inputMatrix,1)):
        for j in range(np.size(inputMatrix,1)):
            f,Pxy = csd(inputMatrix[:,i],inputMatrix[:,j],sampleFreq, nperseg=nfft)
            PSD_matrix[i,j,:] = Pxy
            freq_matrix[i,j,:] = f
            
    
    #Allocating space
    testMat = PSD_matrix[:,:,1]
    testSVD = svd(testMat)
    u_svd = testSVD[0]
    
    
    s1 = np.empty((np.size(PSD_matrix,2),cols),dtype='float')
    ms = np.empty((np.size(u_svd,0),np.size(u_svd,1),np.size(PSD_matrix,2)),dtype='complex64')
    
    #Performing Singular-value decomposition on the PSD-matrix.
    #By default, based on the assumption that the vibration of frequency k is dominated by a single mode, 
    #only the first and most prominent singular value, s1 ,is collected. The mode shape corresponding to s1 is collected in ms.
    for i in range(np.size(PSD_matrix,2)):
        u,s,vh = svd(PSD_matrix[:,:,i])
        s1[i,:] = s[:]
        ms[:,:,i] = u
        
    
    #Creating array of magnitudes in decibel.
    dbs1 = np.empty(np.shape(s1),dtype='float')
    for i in range(np.size(s1,0)):
        dbs1[i,:] = 5*np.log10(np.abs(s1[i,:]))
   
    #Simple peak identification. If a value is larger than both its neighbours, it is determined to be a peak.
    maxList =[]
    maxList_pos = []
    for i in range(1,np.size(s1,0)-1):
        if(s1[i-1,0] < s1[i,0] and s1[i+1,0]<s1[i,0] and s1[i,0] >= peakThresh):
            
            maxList.append(s1[i,0])
            maxList_pos.append(i)
            
    peakFreq = Frequencies[maxList_pos]
    peakMag = s1[maxList_pos] 
    
    
    #Mark peak with both circle and number, where the number can be used later for peak identification.
    xs = peakFreq
    ys = peakMag
    index = []
    for i in range(1,len(ys)+1):
        index.append(i)

    maxListDB =[]
    maxList_posDB = [] #This is a list of the index i at where the max frequencies are identified.
    for i in range(1,np.size(dbs1,0)-1):
        
        if(dbs1[i-1,0] < dbs1[i,0] and dbs1[i+1,0]<dbs1[i,0] and dbs1[i,0] >= peakThresh and Frequencies[i]< frequencyThresh):
            maxListDB.append(dbs1[i,0])
            
            maxList_posDB.append(i)
            
     
    
    peakFreqDB_globalIndex = np.empty((len(maxList_posDB),2),dtype=float)
    peakMagDB = np.empty(len(maxList_posDB),dtype=float)   
    peakMS = np.empty((len(ms),len(maxList_posDB)),dtype = complex)
       
    for i in range(len(maxList_posDB)):
        peakFreqDB_globalIndex[i,0] = maxList_posDB[i]
        peakFreqDB_globalIndex[i,1] = Frequencies[maxList_posDB[i]] #PeakfreqDb now contain the primary [globalIndex,thatpeakFrequency]
        
        peakMagDB[i] = dbs1[maxList_posDB[i],0] 
        peakMS[:,i] = ms[:,0,maxList_posDB[i]]
    
    
    peakFrequencies = peakFreqDB_globalIndex[:,1]

    indexDB = []
    for i in range(1,len(peakMagDB)+1):
        indexDB.append(i)    
    
    peakFreqDataframe = pd.DataFrame(peakFrequencies,columns=['Frequencies[Hz]'])
    peakFreqDataframe.index +=1
    peakFreqDataframe.index.name = "#"
    peakFreqDataframe.columns.name = 'Peak no.'
    
    
    #Plot plot of the 1st singular values
    suggestedValuesPlotTitle =  FDDsolverTitle + "_Suggested 1st singular values, peak treshold = " + str(peakThresh)
    fig,ax = plt.subplots()
    ax.plot(Frequencies,dbs1)
    #for i in range(len(peakFrequencies)):
        #plt.text(peakFrequencies[i], peakMagDB[i], '%s' % indexDB[i] ,size = 12)
    sc = ax.scatter(Frequencies[maxList_posDB],dbs1[maxList_posDB,0])
    ax.set_xlim([np.min(Frequencies),np.max(Frequencies)])
    mplcursors.cursor(sc)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Singular Values [dB]')
    plt.grid()
    plt.show()
    
    

    
    return PSD_matrix,s1,ms,Frequencies
    

def FDDpeak(PSD_matrix, s1, ms, Frequencies, potfrequencies, peakThresh=-10000, frequencyThresh = 60):
    # Frequency Domain Decomposition
    # Input:
    #     inputMatrix        -    [m,n] matrix where each column n contains m sample measurements from one channel
    #     sampleFreq         -    Number of samples per second in Hz
    #     peakThresh         -    Only the peaks with values above this threshold will be enumerated.
    #
    # Output: 
    #     Frequencies        -    Array of frequencies corresponding to the FDD-plot
    #     dbs1               -    Array of the absolute value of the identified singular values given in decibel.
    #     chosenPeaksFreq    -    Array of the frequencies corresponding to the peaks chosen by the user
    #     chosenPeaksMag     -    Array of the magnitude of the peaks chosen by the user
    #     chosenPeaksMS      -    Matrix [n,k] containing k modeshapes. Each row in n corresponds to the channel position n in inputMatrix.
    
    
    
    #Allocating space
        
    
    #Creating array of magnitudes in decibel.
    dbs1 = np.empty(np.shape(s1),dtype='float')
    for i in range(np.size(s1,0)):
        dbs1[i,:] = 5*np.log10(np.abs(s1[i,:]))
   
    #Simple peak identification. If a value is larger than both its neighbours, it is determined to be a peak.
    maxList =[]
    maxList_pos = []
    for i in range(1,np.size(s1,0)-1):
        if(s1[i-1,0] < s1[i,0] and s1[i+1,0]<s1[i,0] and s1[i,0] >= peakThresh):
            
            maxList.append(s1[i,0])
            maxList_pos.append(i)
            
    peakFreq = Frequencies[maxList_pos]
    peakMag = s1[maxList_pos] 
    
    
    #Mark peak with both circle and number, where the number can be used later for peak identification.
    xs = peakFreq
    ys = peakMag
    index = []
    for i in range(1,len(ys)+1):
        index.append(i)

    maxListDB =[]
    maxList_posDB = [] #This is a list of the index i at where the max frequencies are identified.
    for i in range(1,np.size(dbs1,0)-1):
        
        if(dbs1[i-1,0] < dbs1[i,0] and dbs1[i+1,0]<dbs1[i,0] and dbs1[i,0] >= peakThresh and Frequencies[i]< frequencyThresh):
            maxListDB.append(dbs1[i,0])
            
            maxList_posDB.append(i)
            
     
    
    peakFreqDB_globalIndex = np.empty((len(maxList_posDB),2),dtype=float)
    peakMagDB = np.empty(len(maxList_posDB),dtype=float)   
    peakMS = np.empty((len(ms),len(maxList_posDB)),dtype = complex)
       
    for i in range(len(maxList_posDB)):
        peakFreqDB_globalIndex[i,0] = maxList_posDB[i]
        peakFreqDB_globalIndex[i,1] = Frequencies[maxList_posDB[i]] #PeakfreqDb now contain the primary [globalIndex,thatpeakFrequency]
        
        peakMagDB[i] = dbs1[maxList_posDB[i],0] 
        peakMS[:,i] = ms[:,0,maxList_posDB[i]]
    
    
    peakFrequencies = peakFreqDB_globalIndex[:,1]

    indexDB = []
    for i in range(1,len(peakMagDB)+1):
        indexDB.append(i)    
    
    peakFreqDataframe = pd.DataFrame(peakFrequencies,columns=['Frequencies[Hz]'])
    peakFreqDataframe.index +=1
    peakFreqDataframe.index.name = "#"
    peakFreqDataframe.columns.name = 'Peak no.'
    
    
    #Plot plot of the 1st singular values
    fig,ax = plt.subplots()
    ax.plot(Frequencies,dbs1)
    ax.set_xlim([np.min(Frequencies),np.max(Frequencies)])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('1st Singular Values [dB]')
    plt.grid()
    plt.show()
    
    peakMagDB_hstack = np.hstack(peakMagDB)
    peakFreqDB_hstack = np.hstack(peakFreqDB_globalIndex[:,1])
    
     # Select peaks
    selectedPeakIndex = np.zeros((len(potfrequencies)))
    cont = 0
    for i in potfrequencies:
         selectedPeakIndex[cont] = find_nearest(peakFreqDB_hstack, i)
         cont = cont+1
    selectedPeakIndex = np.sort(selectedPeakIndex)
    selectedPeakIndex=selectedPeakIndex.astype(dtype=int)
    
    chosenPeaksFreq = np.empty(len(selectedPeakIndex),dtype=float)
    chosenPeaksMag = np.empty(len(selectedPeakIndex),dtype = float)
    chosenPeaksMS = np.empty((len(peakMS),len(selectedPeakIndex)),dtype = complex)
    
    
    for i in range(len(selectedPeakIndex)):
        chosenPeaksFreq[i] = peakFreqDB_hstack[selectedPeakIndex[i]]
        chosenPeaksMag[i] = peakMagDB_hstack[selectedPeakIndex[i]]
        chosenPeaksMS[:,i] = peakMS[:,selectedPeakIndex[i]]
    
    
    #Compute mode shapes
    outCols = np.size(chosenPeaksMS,1)
    outRows = (np.size(chosenPeaksMS,0)+2)
    
    for i in np.arange(len(chosenPeaksFreq)):
     plt.plot(chosenPeaksFreq[i], chosenPeaksMag[i],'ro')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Singular values [dB]')
    plt.grid()
    plt.show()
    
    return Frequencies, dbs1, chosenPeaksFreq, chosenPeaksMag, chosenPeaksMS



def FDDdamping(PSD_matrix,s1,ms,Frequencies,samplingFreq,ResFreq,MOD,deltafreq=0.5):
    
    df = Frequencies[1]-Frequencies[0]
    Ntake = int((deltafreq/df)/2)
    
    # Select Portions of SV
    [gr, gc, nk] = np.shape(PSD_matrix);
    ntw = int(np.floor(nk/2))
    PSD = PSD_matrix[:,:,0:2*ntw+1]
    [gr, gc, nk] = np.shape(PSD);

    nm = len(ResFreq)
    Gm = np.zeros((nm, nk)).astype(dtype=float)
    damping = np.zeros((nm, 1))
    Vp = np.linalg.pinv(MOD).T
    for m in np.arange(0,nm,1):
          pos = find_nearest(Frequencies, ResFreq[m])
          for k in np.arange(pos-Ntake,pos+Ntake,1):
             Gm[m,k] = np.real(np.dot(np.dot(Vp[:,m].T,PSD[:,:,k]),Vp[:,m]));
    G = np.concatenate((Gm, np.conj(np.fliplr(Gm[:,1:-1]))),axis=1)
    R = ifft(G.T,axis=0)
    R1 = R[0:nk,:]
    dt = 1/samplingFreq
    
    for m in np.arange(0,nm,1):
        plt.subplot(1,nm,m+1)
        T = 50./ResFreq[m]
        npp = int(T/dt)
        hilsig = hilbert(np.real(R1[1:npp,m].T))
        envelope = np.abs(hilsig)
        tvect = np.zeros((npp-1,1))
        tvect[:,0] = np.arange(0,npp-1,1)*dt
        Amp = np.real(R1[0,m])
        nelim = int((3./ResFreq[m])//dt)
        model = np.polyfit(tvect[nelim:-nelim,0], np.log(envelope[nelim:-nelim]/Amp), 1)
        chi = -model[0]/(2*np.pi*ResFreq[m])
        modpred = Amp*np.exp(-chi*2*np.pi*ResFreq[m]*tvect)*np.cos(2*np.pi*ResFreq[m]*tvect)
        plt.plot(tvect,np.real(R1[1:npp,m]),'r')
        plt.plot(tvect[nelim:-nelim],np.abs(envelope[nelim:-nelim]),'--b')
        plt.plot(tvect,modpred,'b')
        damping[m] = chi*100
    

    
    
    return damping


# ***********************************************************************************************
# ***********************************************************************************************
# ***********************************************************************************************
#                                      COV-SSI
# ***********************************************************************************************
# ***********************************************************************************************
# ***********************************************************************************************
# ***********************************************************************************************


def covssi(data, fs, i, orders, weighting='none', matrix_type='hankel', algorithm='shift', showinfo=False, balance=True):
    """
    Main function for covariance-driven SSI.

    Arguments
    ---------------------------
    data : double
        data matrix, with channels stacked column-wise (n_samples-by-n_channels)
    fs : double
        sampling frequency
    i : int
        maximum number of block rows
    orders : int
        array or list of what orders to include
    weighting : 'none', optional
        what weighting type to use ('none' or 'br', or 'cva')
    matrix_type : 'matrix_type', optional
        what matrix type formulation to base computation on ('hankel' or 'toeplitz')
    algorithm : 'shift', optional
        what algorithm to use ('shift' or 'standard' - when using 'standard' the method is equivalent to NExt-ERA)
    showinfo : True, optional
        whether or not to print information during computation
    balance : True, optional
        whether or not to conduct balancing to the cross-correlation matrices prior to matrix operations (Cholesky and SVD)

    Returns
    ---------------------------
    lambd : double
        list of arrays with complex-valued eigenvalues (each list entry correspond to one order)
    phi : double
        list of 2d arrays with complex-valued eigenvectors (each list entry correspond to one order), each column corresponds to a mode

    References
    --------------------------------
    * Hermans and van Der Auweraer :cite:`Hermans1999` (1 in code)
    * Van Overschee and de Moor :cite:`VanOverschee1996` (2 in code)
    * Rainieri and Fabbrocino :cite:`Rainieri` (3 in code)
    * Pridham and Wilson :cite:`Pridham2003` (4 in code)
    """ 

    if not (weighting in ['cva', 'br', 'none']):
        raise ValueError('Invalid weighting algorithm requested ("cva", "br", "none" accepted)')

    if showinfo:
        print('*** Covariance-driven SSI algorithm for OMA ***')

    dt = 1.0/fs
    l = np.shape(data)[1]

    # Establish Toeplitz/Hangel and lag-shifted Toeplitz/Hankel matrix [1]
    if showinfo:
        print('> Establishing Hankel/Toeplitz matrices')
        print('  >> Correlation estimation')

    R = xcorr_lag(data, 2*i+1)  #including no time lag entry, R_0 = R[:,:,0] - such that R_1-->R_2i+1 & R_0 makes 2i+2 in total

    # Matrix stacking
    if showinfo:
        print('  >> Matrix stacking')

    if matrix_type == 'hankel':
        H0, H1 = stack_hankel(R)
    elif matrix_type == 'toeplitz':
        H0, H1 = stack_toeplitz(R)
    else:
        raise ValueError('Unknown matrix string.')

    # Matrix stacking
    # Weighting
    W1, W2, L1, L2 = [np.eye(i*l)]*4 # Standard values

    if showinfo:
        print('> Establishing weighting matrices')
        print('  >> Weighting requested: %s' % (weighting.upper()))

    if weighting == 'cva':  #[2] used, [1] results in ill-conditioning issues
        try:
            if showinfo:
                print('  >> Establishing R+ and R-')
            
            W1, W2, L1, L2 = cva_weights(R, balance=balance)
            
            if algorithm == 'standard':
                print('Solution algorithm STANDARD is not supported with CVA. Changing to SHIFT algorithm.')
                algorithm='shift'
        except:
           print('  >> CVA failed! Continuing with no weighting (BR). If you want to run with CVA, try adjusting number of block rows.')

    if showinfo:
        print('> Computing SVD')
 
    U, d, Q = np.linalg.svd(W1 @ H0 @ W2.T)
    V = Q.T     # comes out transposed from svd function
    D = np.diag(d)  # make matrix

    # Eigenvalue solution
    if showinfo:
        print('> Computing state matrix for each order to establish modes')

    lambd = [None]*len(orders)
    phi = [None]*len(orders)

    for j in range(0, len(orders)):
        o = orders[j]

        # Select subset from H0, via the SVD decomposition, based on order
        D1 = D[0:o, :][:, 0:o]
        sqrtD1 = np.sqrt(D1)
        U1 = U[:, 0:o]
        V1 = V[:, 0:o]
        
        # State matrix estimate
        O = L1 @ U1 @ sqrtD1   #O, observability (T=identity matrix)
        C = O[0:l, :]      #Pick the l first values
        
        Oup = O[0:-l, :]   #Remove the last l rows from O
        Odown = O[l:, :]   #Remove the first l rows from O

        if algorithm == 'standard': #this means NExT-ERA is chosen
            A = np.linalg.inv(sqrtD1) @ U1.T @ H1 @ V1 @ np.linalg.inv(sqrtD1)        # [4] (and [3])
        elif algorithm == 'shift':
            A = np.linalg.pinv(Oup) @ Odown  # [1]

        # Eigenvalue decomposition [1] and convertion from discrete to continuous poles
        lambd_d, psi_d = np.linalg.eig(A)        #system eigenvectors and eigenvalues
        
        lambd_j = np.log(lambd_d)/dt    #make vector from diagonal matrix and transform from discrete to continuous
        phi_j = C @ psi_d                          #observed part of system eigenvectors, referring to input channels
        
        # Sort and order modes 
        sortix = np.argsort((np.abs(lambd_j)))      #find index for unique absolute values
        lambd[j] = lambd_j[sortix]                   #keep the corresponding eigenvalues
        phi[j] = phi_j[:, sortix]                    #and the corresponding eigenvectors

    if showinfo:
        print('> Computation completed')

    return lambd, phi



def xcorr_lag(data, maxlag):
    """
    Computes cross-correlation matrices for multiple time lags based on input data.

    Arguments
    ---------------------------
    data : double
        matrix with data, n_samples-by-n_channels (channels are column-wise)
    maxlag : int
        maximum number of sample lags

    Returns
    ---------------------------
    R : double
        n_channels-by-n_channels-by-n_lags large array, each slice in third dimension 
        corresponds to the cross-correlation matrix for a given time lag 

    """

    l = np.shape(data)[1]
    nsamples = np.shape(data)[0]

    if maxlag>(nsamples-1):
        print('Maximum sample lag is larger than total sample length! Reducing to correspond to one below sample length.')
        maxlag = nsamples-1
    
    R0 = np.zeros([l, l, maxlag+1])
    
    for dof1 in range(0, l):
        for dof2 in range(0, l):
            R0[dof1, dof2, :] = correlate(data[:, dof2], data[:,dof1], mode='full', method='auto')[nsamples-1:nsamples+maxlag]    #matches MATLAB implementation - verify. dof1, dof2 and [(nsamples+1):(nsamples+1-maxlag):-1] looks more correct.
            
    unbiasing_scaling = (nsamples - np.linspace(0, maxlag, maxlag+1))[np.newaxis,np.newaxis,:]
    R = R0/unbiasing_scaling    # ensure unbiased estimate

    return R


def stack_hankel(R):
    """
    Stacks lag-shifted cross-correlation matrices (stacked along third dimension) into Hankel matrix.

    Arguments
    ---------------------------
    R : double
        n_channels-by-n_channels-by-n_lags large array, each slice in third dimension 
        corresponds to the cross-correlation matrix for a given time lag 

    Returns
    ---------------------------
    H0 : double
        block-Hankel matrix
    H1 : double
        one-lag shifted block-Hankel matrix

    References
    --------------------------------
    Hermans and van Der Auweraer :cite:`Hermans1999`

    """

    i = int(np.shape(R)[2]/2)-1
    l = np.shape(R)[0]

    H0 = np.zeros([l*i, l*i])
    H1 = np.zeros([l*i, l*i])

    for row in range(0, i):             #go through all block rows from 1 to i
            for col in range(0, i):         #go through all block columns from 1 to i
                H0[(row*l):(row*l+l), (col*l):(col*l+l)] = R[:, :, row+col+1]           #Hankel matrix, H0
                H1[(row*l):(row*l+l), (col*l):(col*l+l)] = R[:, :, row+col+2]           #Lag-shifted Hankel matrix, H1    

    return H0, H1


def stack_toeplitz(R):
    """
    Stacks lag-shifted cross-correlation matrices (stacked along third dimension) into Toeplitz matrix.

    Arguments
    ---------------------------
    R : double
        n_channels-by-n_channels-by-n_lags large array, each slice in third dimension 
        corresponds to the cross-correlation matrix for a given time lag 

    Returns
    ---------------------------
    H0 : double
        block-Toeplitz matrix
    H1 : double
        one-lag shifted block-Toeplitz matrix

    References
    --------------------------------
    Rainieri and Fabbrocino :cite:`Rainieri`

    """

    i = int(np.shape(R)[2]/2)-1
    l = np.shape(R)[0]

    H0 = np.zeros([l*i, l*i])
    H1 = np.zeros([l*i, l*i])

    for row in range(0, i):             #go through all block rows from 1 to i
        for col in range(0, i):         #go through all block columns from 1 to i
            H0[(row*l):(row*l+l), (col*l):(col*l+l)] = R[:, :, i+1-col+row]            # Toeplitz matrix, H0
            H1[(row*l):(row*l+l), (col*l):(col*l+l)] = R[:, :, i+1-col+row+1]       

    return H0, H1

def cva_weights(R, balancing_H0=None, balance=False):
    """
    Computes the weights for CVA.

    Arguments
    ---------------------------
    R : double
        n_channels-by-n_channels-by-n_lags large array, each slice in third dimension 
        corresponds to the cross-correlation matrix for a given time lag 
    balancing_H0 : boolean, optional
    balance : boolean, optional

    Returns
    ---------------------------
    W1 : double
        lower triangular Cholesky factor of R+ (see Hermans and van der Auweraer)
    W2 : double
        lower triangular Cholesky factor of R- (see Hermans and van der Auweraer)
    L1 : double
        inverse of W1
    L2 : double
        inverse of W2

    References
    --------------------------------
    Hermans and van Der Auweraer :cite:`Hermans1999`

    """

    i = int(np.shape(R)[2]/2)-1
    l = np.shape(R)[0]

    Rp = np.zeros([i*l, i*l])
    Rm = np.zeros([i*l, i*l])

    for row in range(0, i):
        for col in range(0, i):
            # R+
            if row<col: #above diagonal
                Rp[(row*l):(row*l+l), (col*l):(col*l+l)] = R[:, :, np.abs(row-col)].T
            elif row>=col:  #below or at diagonal
                Rp[(row*l):(row*l+l), (col*l):(col*l+l)] = R[:, :, np.abs(row-col)]

            # R-
            if row<=col:
                Rm[(row*l):(row*l+l), (col*l):(col*l+l)] = R[:, :, np.abs(row-col)]
            elif row>col:
                Rm[(row*l):(row*l+l), (col*l):(col*l+l)] = R[:, :, np.abs(row-col)].T   

    if balancing_H0 is not None:
        __, T = matrix_balance(balancing_H0)
        Rp = np.linalg.inv(T) @ Rp @ T
        Rm = np.linalg.inv(T) @ Rm @ T
    
    if balance:
        Rp, __ = matrix_balance(Rp)
        Rm, __ = matrix_balance(Rm)

    L1 = gaxpy_chol(Rp)            
    L2 = gaxpy_chol(Rm)
    W1 = np.linalg.inv(L1)
    W2 = np.linalg.inv(L2)

    return W1, W2, L1, L2



def find_stable_poles(lambd, phi, orders, s, stabcrit={'freq': 0.05, 'damping': 0.1, 'mac': 0.1}, valid_range={'freq': [0, np.inf], 'damping':[0, np.inf]}, indicator='freq'):
    """
    Post-processing of Cov-SSI results, to establish modes (stable poles) from all poles.

    Arguments
    ---------------------------
    lambd : double
        list of arrays with complex-valued eigenvalues, one list entry per order
    phi : double
        list of 2d arrays with complex-valued eigevectors (stacked column-wise), one list entry per order
    orders : int
        orders corresponding to the list entries in lambd and phi
    s : int
        stability level, see :cite:`Kvale2017_OMA`
    stabcrit : {'freq': 0.05, 'damping':0.1, 'mac': 0.1}, optional
        criteria to be fulfilled for pole to be deemed stable
    valid_range : {'freq': [0, np.inf], 'damping':[0, np.inf]}, optional
        valid ranges of frequencies (rad/s) and damping for pole to be deemed stable
    indicator : 'freq', optional
        what modal indicator to use ('freq' or 'mac')

    Returns
    ---------------------------
    lambd_stab : double
        array with complex-valued eigenvalues deemed stable
    phi_stab : double
        2d array with complex-valued eigenvectors (stacked as columns), each column corresponds to a mode
    orders_stab : int
        corresponding order for each stable mode
    idx_stab : int
        indices of poles (within its order from the input) given in lambd_stab deemed stable (for indexing post)
        
    References
    --------------------------------
    Kvåle et al. :cite:`Kvale2017_OMA`

    """ 

    wtol = stabcrit['freq']
    xitol = stabcrit['damping']
    mactol = stabcrit['mac']

    if 'freq' not in valid_range.keys():
        valid_range['freq'] = [0, np.inf]
    
    if 'damping' not in valid_range.keys():
        valid_range['damping'] = [0, np.inf]

    lambd_stab = []
    phi_stab = []
    orders_stab = []
    idx_stab = []
    
    # Establish for all orders above stablevel
    for order_ix in range(s-1, len(orders)):   # i is order
        omega = abs(lambd[order_ix])
        xi = -np.real(lambd[order_ix])/abs(lambd[order_ix])
        phi_this = phi[order_ix]    # modal transformation matrix for order with index i
    
        # Stable poles
        for pole_ix in range(0, len(omega)):
            stab = 0
            for level in range(1, s+1):
                phi_last = phi[order_ix-level]
    
                if indicator == 'mac':
                    macs = xmacmat_alt(phi_last, phi_this[:, pole_ix], conjugates=True)
                    pole_ix_last = np.argmax(macs[:,0])   #find largest mac in first column (all in matrix compared with vector)
                elif indicator == 'freq':
                    omega_last = abs(lambd[order_ix-level])
                    pole_ix_last = np.argmin(abs(omega[pole_ix]-omega_last))
    
                lambd_last = lambd[order_ix-level][pole_ix_last]
                xi_last = -np.real(lambd_last)/abs(lambd_last)
                omega_last = abs(lambd_last)
                dxi = abs(xi[pole_ix] - xi_last)
                dw = abs(omega[pole_ix] - omega_last)
                mac = xmacmat_alt(phi_last[:, pole_ix_last], phi_this[:, pole_ix], conjugates=True)
    
                if ((dw/omega_last)<=wtol) and ((dxi/xi_last)<=xitol) and (mac>=(1-mactol)) and (valid_range['freq'][0]<omega_last<valid_range['freq'][1]) and (valid_range['damping'][0]<xi_last<valid_range['damping'][1]):
                    stab += 1
                else:
                    stab = 0
                    break

            if stab>=s:
                lambd_stab.append(lambd[order_ix][pole_ix])
                phi_stab.append(phi[order_ix][:, pole_ix])
                orders_stab.append(orders[order_ix])
                idx_stab.append(pole_ix)
                
    phi_stab = np.array(phi_stab).T
            
    return lambd_stab, phi_stab, orders_stab, idx_stab



def xmacmat_alt(phi1, phi2=None, conjugates=True):
    """
    Alternative implementation. Modal assurance criterion numbers, cross-matrix between two modal transformation matrices (modes stacked as columns).

    Arguments
    ---------------------------
    phi1 : double
        reference modes
    phi2 : double, optional
        modes to compare with, if not given (i.e., equal default value None), phi1 vs phi1 is assumed
    conjugates : True, optional
        check the complex conjugates of all modes as well (should normally be True)

    Returns
    ---------------------------
    macs : boolean
        matrix of MAC numbers
    """

    if phi2 is None:
        phi2 = phi1
    
    if phi1.ndim==1:
        phi1 = phi1[:, np.newaxis]
    
    if phi2.ndim==1:
        phi2 = phi2[:, np.newaxis]
    
    A = np.sum(phi1.T * np.conj(phi1.T), axis=1)[:,np.newaxis]
    B = np.sum(phi2.T * np.conj(phi2.T), axis=1)[:, np.newaxis]
    norms = np.abs(A @ B.T)

    if conjugates:
        macs = np.maximum(abs(phi1.T @ phi2)**2/norms, abs(phi1.T @ phi2)**2/norms)     #maximum = element-wise max
    else:
        macs = abs(phi1.T @ phi2)**2/norms
    
    macs = np.real(macs)
    
    return macs

def Clustering(lambd_stab, phi_stab):
      
    omega = np.abs(np.imag(lambd_stab))
    fn0 = omega/(2*np.pi)
    zeta0 = -np.real(lambd_stab)/np.abs(lambd_stab)
    phi0 = phi_stab
    Nsamples = np.size(lambd_stab);
    pos = np.zeros((Nsamples,Nsamples))
    for i1 in np.arange(0,Nsamples):
        for i2 in np.arange(0,Nsamples):
            MAC0 = MAC(phi0[:,i1],phi0[:,i2]); 
            pos[i1,i2] = np.abs((fn0[i1]-fn0[i2])/fn0[i2]) +1-MAC0

        

    Z =  linkage(pos,method='single',metric='euclidean');
    myClus = fcluster(Z,0.4,criterion='distance')
    Ncluster = np.max(myClus)
    ss=1
    fn = np.zeros((Ncluster,np.size(phi_stab,1)))
    zeta = np.zeros((Ncluster,np.size(phi_stab,1)) )
    phi = np.zeros((np.size(phi_stab,0),Ncluster,np.size(phi_stab,1)))
    phi = phi.astype(dtype = 'complex')
    for rr in np.arange(1,Ncluster+1):
          if len(myClus[myClus==rr])>5:
               dummyZeta = zeta0[myClus==rr]
               dummyFn = fn0[myClus==rr]
               dummyPhi = phi0[:,myClus==rr]
               valMin = np.max(np.array(0,(np.quantile(dummyZeta,0.25) - np.abs(np.quantile(dummyZeta,0.75)-np.quantile(dummyZeta,0.25))*1.5)))
               valMax = np.quantile(dummyZeta,0.75) + np.abs(np.quantile(dummyZeta,0.75)-np.quantile(dummyZeta,0.25))*1.5
               dummyFn[np.logical_or(dummyZeta>valMax,dummyZeta<valMin)] = np.NaN;
               dummyPhi[:,np.logical_or(dummyZeta>valMax,dummyZeta<valMin)] = np.NaN;
               dummyZeta[np.logical_or(dummyZeta>valMax,dummyZeta<valMin)] = np.NaN;
               fn[ss,0:len(dummyFn)] = dummyFn;
               zeta[ss,0:len(dummyFn)] = dummyZeta;
               phi[:,ss,0:len(dummyFn)] = dummyPhi;
               ss=ss+1;
    fn[fn == 0] = 'nan'  
    zeta[zeta == 0] = 'nan' 
    phi[phi == 0] = 'nan'  
    Clustersize = np.sum(~np.isnan(fn), 1) 
    fn = np.nanmean(fn.T,axis=0)
    zeta = 100*np.nanmean(zeta.T,axis=0)
    phi = np.nanmean(phi,axis=2)
    
    fn = fn[~np.isnan(fn)]
    zeta = zeta[~np.isnan(zeta)]
    phi = phi[:,~np.isnan(phi[0,:])]
    Clustersize = Clustersize[np.nonzero(Clustersize)]
    
    

    return [fn,zeta,phi,Clustersize]  



def plotframemodes(mult, MOD, lh, lv, Expfreq):
    plt.clf()
    xun,yun = convert_mode(MOD[:,0]*0, lh, lv)
    MOD = MOD*mult
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