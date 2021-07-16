# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 09:42:27 2021

@author: Enrique GM
"""

import numpy as np
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate as interp
from matplotlib.colors import LightSource


class getShapeFunctionStructure():        
    def shapeFunction(self, xi, eta):
        self.shape=(1.0/4.0)*np.array([[
        (1-xi)*(1-eta)],[(1+xi)*(1-eta)],
        [(1+xi)*(1+eta)],[(1-xi)*(1+eta)]])
        self.der=(1.0/4.0)*np.array([[-(1-eta), -(1-xi)],[1-eta, -(1+xi)],[
        1+eta, 1+xi],[-(1+eta), 1-xi]])
        
class JacobianStructure():   
        def Jacobian_matrix(self,nodeCoordinates,naturalDerivatives):
            self.matrix=np.dot(nodeCoordinates.T,naturalDerivatives)
            self.inv=np.linalg.inv(self.matrix)
            self.derivatives=np.dot(naturalDerivatives,self.inv)
            # Jac.matrix : Jacobian matrix
            # Jac.inv : inverse of Jacobian Matrix
            # Jac.derivatives : derivatives w.r.t. x and y
            # naturalDerivatives : derivatives w.r.t. xi and eta
            # nodeCoordinates : nodal coordinates at element level

def isotropic(Ey,poism):
  
  # Compute constitutive tensor of an isotropic material
  # INPUT:
  # Ey -> Young's Modulus
  # poism -> Poisson's ration
  # OUTPUT:
  # C -> Constitutive isotropic tensor 6x6
  
  # Author: E. García-Macías
  # -----------------------------------------------------------------------


  G=Ey/(2*(1+poism));
  C=np.zeros((6,6))
  C[0,:]=[Ey*(1-poism)/((1-2*poism)*(1+poism)),Ey*poism/((1-2*poism)*(1+poism)),Ey*poism/((1-2*poism)*(1+poism)),0,0,0];
  C[1,:]=[Ey*poism/((1-2*poism)*(1+poism)),Ey*(1-poism)/((1-2*poism)*(1+poism)),Ey*poism/((1-2*poism)*(1+poism)),0,0,0];
  C[2,:]=[Ey*poism/((1-2*poism)*(1+poism)),Ey*poism/((1-2*poism)*(1+poism)),Ey*(1-poism)/((1-2*poism)*(1+poism)),0,0,0];
  C[3,:]=[0,0,0,G,0,0];
  C[4,:]=[0,0,0,0,G,0];
  C[5,:]=[0,0,0,0,0,G];

  return C  


def Planestress(C):

  S=np.linalg.inv(C)

  S = np.delete(S, 2, 0)
  S = np.delete(S, 2, 1)

  Cf=np.array([[S[0,0],S[0,1],S[0,4]],[
      S[1,0],S[1,1],S[1,4]],[
      S[4,0],S[4,1],S[4,4]]])

  Cs=np.array([[S[2,2],S[2,3]],[
    S[3,2],S[3,3]]])

  Cf=np.linalg.inv(Cf);
  Cs=np.linalg.inv(Cs);

  return [Cf, Cs];




def ElemCoordxy(ndivx,ndivy,a,b):
  nel=-1
  Coordelemx=np.zeros((ndivx*ndivy,4))
  Coordelemy=np.zeros((ndivx*ndivy,4))
  for i in np.arange(1,ndivy+1,1):
    for j in np.arange(1,ndivx+1,1):
        nel=nel+1
        Coordelemx[nel,0]=(a/ndivx)*(j-1)
        Coordelemx[nel,1]=(a/ndivx)+(a/ndivx)*(j-1)
        Coordelemx[nel,2]=(a/ndivx)+(a/ndivx)*(j-1)
        Coordelemx[nel,3]=(a/ndivx)*(j-1)
        Coordelemy[nel,0]=(b/ndivy)*(i-1)
        Coordelemy[nel,1]=(b/ndivy)*(i-1)
        Coordelemy[nel,2]=(b/ndivy)+(b/ndivy)*(i-1)
        Coordelemy[nel,3]=(b/ndivy)+(b/ndivy)*(i-1)
  return [Coordelemx, Coordelemy];  



def ElemTipology(ndivx,ndivy,ndof):

  Edof=np.zeros((ndivx*ndivy,4*ndof+1))
  cont=-1
  nodos=np.zeros((ndivx*ndivy,4));

  for i in np.arange(1,ndivy+1,1):
    for j in np.arange(1,ndivx+1,1):
        cont=cont+1
        nodos[cont,0]=(i-1)*(ndivx+1)+j
        nodos[cont,1]=(i-1)*(ndivx+1)+j+1
        nodos[cont,2]=nodos[cont,1]+ndivx+1
        nodos[cont,3]=nodos[cont,0]+ndivx+1
 
  for i in np.arange(0,ndivx*ndivy,1):
    nodo1=nodos[i,0]
    nodo2=nodos[i,1]
    nodo3=nodos[i,2]
    nodo4=nodos[i,3]

    Edof[i,0]=i+1
    Edof[i,1:ndof+1]=np.arange((nodo1-1)*ndof+1,nodo1*ndof+1,1)
    Edof[i,ndof+1:2*ndof+1]=np.arange((nodo2-1)*ndof+1,nodo2*ndof+1,1)
    Edof[i,2*ndof+1:3*ndof+1]=np.arange((nodo3-1)*ndof+1,nodo3*ndof+1,1)
    Edof[i,3*ndof+1:4*ndof+1]=np.arange((nodo4-1)*ndof+1,nodo4*ndof+1,1)

  return [Edof, nodos];  


def assem(Edof,K,Ke,M,Me,f,fe):
# K=assem(edof,K,Ke)
# [K,f]=assem(edof,K,Ke,f,fe)
#-------------------------------------------------------------
# PURPOSE
#  Assemble element matrices Ke ( and fe ) into the global
#  stiffness matrix K ( and the global force vector f )
#  according to the topology matrix edof.
#
# INPUT: edof:       dof topology matrix
#       K :         the global stiffness matrix
#        Ke:         element stiffness matrix
#        Me:         element mass matrix
#        f :         the global force vector
#        fe:         element force vector
#
# OUTPUT: K :        the new global stiffness matrix
#         M :        the new global mass matrix
#         f :        the new global force vector
#-------------------------------------------------------------
#-------------------------------------------------------------

  nn=len(Edof)
  eftab=Edof

  for i in np.arange(0,nn,1):
    ii=int(eftab[i])-1
    f[0,ii]=f[0,ii]+fe[i]
    for j in np.arange(i,nn,1):
        jj=int(eftab[j])-1
        K[ii,jj]=K[ii,jj]+Ke[i,j]
        K[jj,ii]=K[ii,jj]
        M[ii,jj]=M[ii,jj]+Me[i,j]
        M[jj,ii]=M[ii,jj]

  return [K,M,f] 



def shell_midlin(ex,ey,ez,ep,h,P):

  Em=ep[0]
  poism=ep[1]
  dens=ep[2]
  ks=ep[3]
  Gm=Em/(1*(1+poism))

  # Constitutive matrix
  #C=ortoenginneer(Em,Em,Em,poism,poism,poism,poism,poism,poism,Gm,Gm,Gm);
  C=isotropic(Em,poism)
  # Plane stress
  [Cf,Cs]=Planestress(C);

  Ce=Cf*h
  Cf=Cf*h**3/12.  
  Cs=Cs*ks*h
  Cc=Cf*0.
  
  
  # Mass integration across thickness
  I1 = dens*h
  I2 = 0.
  I3 = dens*h**3/12
  MI=np.array([[I1 , 0.   ,  0. ,   I2   , 0.],
    [0. , I1  ,  0. ,  0.  , I2],
    [0.  ,  0.  , I1 ,   0.   ,  0.],
    [I2  ,  0.  ,  0. , I3  , 0.],
    [0.  ,  I2   ,  0. ,  0.  , I3]])
  
  
  # Gauss points and weights
  dimelem = 2
  nOfGaussPoints = 2
  # [z,w] = gaussLegendre(nOfGaussPoints,-1,1);
  z=np.array([-0.577350269189626,0.577350269189626])
  w=np.array([1.,1.])
  # theReferenceElement.gaussPoints = z;
  # theReferenceElement.gaussWeights = w;
  
  
  # Construction of the elemental stiffness matrix
  Ke=np.zeros((20,20))
  Me=np.zeros((20,20))
  fe=np.zeros((20,1))
  
  ndof=4
  B_m=np.zeros((3,5*ndof))
  B_b=np.zeros((3,5*ndof))
  B_s=np.zeros((2,5*ndof))
  Nsh=np.zeros((5,5*ndof))
  
  
  for j in np.arange(0,dimelem,1):
    for i in np.arange(0,nOfGaussPoints,1):
        
        # Bending
        # ------------------- Shape functions at Gauss points
        shapeFun = getShapeFunctionStructure()
        shapeFun.shapeFunction(z[i],z[j])
        # ------------------- Jacobian
        Jac=JacobianStructure()
        Jac.Jacobian_matrix(np.vstack((ex,ey)).T,shapeFun.der)
        
        # [B] matrix membrane
        B_m[0,0:0+4*5:5] = Jac.derivatives[:,0].T
        B_m[1,1:1+4*5:5]= Jac.derivatives[:,1].T
        B_m[2,0:0+4*5:5] = Jac.derivatives[:,1].T
        B_m[2,1:1+4*5:5]= Jac.derivatives[:,0].T
        
        # [B] matrix bending
        B_b[0,3:3+4*5:5] = Jac.derivatives[:,0].T
        B_b[1,4:4+4*5:5] = Jac.derivatives[:,1].T
        B_b[2,3:3+4*5:5] = Jac.derivatives[:,1].T
        B_b[2,4:4+4*5:5] = Jac.derivatives[:,0].T
        
        
        BE=np.dot(np.dot(B_m.T,Ce),B_m)*w[i]*w[j]*np.linalg.det(Jac.matrix)
        BB=np.dot(np.dot(B_b.T,Cf),B_b)*w[i]*w[j]*np.linalg.det(Jac.matrix)
        # COUPLING
        BC=np.dot(np.dot(B_m.T,Cc),B_b)*w[i]*w[j]*np.linalg.det(Jac.matrix)
        BC=BC+np.dot(np.dot(B_b.T,Cc),B_m)*w[i]*w[j]*np.linalg.det(Jac.matrix)
        
        
        # MASS
        Nsh[0,0:0+4*5:5]=shapeFun.shape.T
        Nsh[1,1:1+4*5:5]=shapeFun.shape.T
        Nsh[2,2:2+4*5:5]=shapeFun.shape.T
        Nsh[3,3:3+4*5:5]=shapeFun.shape.T
        Nsh[4,4:4+4*5:5]=shapeFun.shape.T
        Me=Me+np.dot(np.dot(Nsh.T,MI),Nsh)*w[i]*w[j]*np.linalg.det(Jac.matrix);
        
        Ke=Ke+BE+BB+BC     
  
  
  # Gauss points and weights
  
  nOfGaussPoints = 1;
  # [z,w] = gaussLegendre(nOfGaussPoints,-1,1);
  z=np.array([0.,0.]);
  w=np.array([2.,2.]);
  
  dimelem=1;
  for j in np.arange(0,dimelem,1):
    for i in np.arange(0,nOfGaussPoints,1):
        # Shear
        # ------------------- Shape functions at Gauss points
        shapeFun = getShapeFunctionStructure()
        shapeFun.shapeFunction(z[i],z[j])
        # ------------------- Jacobian
        Jac=JacobianStructure()
        Jac.Jacobian_matrix(np.vstack((ex,ey)).T,shapeFun.der)
        
        # [B] matrix shear
        B_s[0,2:2+4*5:5] = Jac.derivatives[:,0].T
        B_s[1,2:2+4*5:5] = Jac.derivatives[:,1].T
        B_s[0,3:3+4*5:5] = shapeFun.shape.T
        B_s[1,4:4+4*5:5]= shapeFun.shape.T
        BS=np.dot(np.dot(B_s.T,Cs),B_s)*w[i]*w[j]*np.linalg.det(Jac.matrix);


    Ke=Ke+BS;  


    # FORCE
    Nsh[0,0:0+4*5:5]=shapeFun.shape.T
    Nsh[1,1:1+4*5:5]=shapeFun.shape.T
    Nsh[2,2:2+4*5:5]=shapeFun.shape.T
    Nsh[3,3:3+4*5:5]=shapeFun.shape.T
    Nsh[4,4:4+4*5:5]=shapeFun.shape.T
    fe=fe+np.dot(Nsh.T,np.array([[0.,0.,P,0.,0.]]).T)*w[i]*w[j]*np.linalg.det(Jac.matrix);


  return [Ke,Me,fe]; 


def SSSS(ndivx,ndivy,ndof):
  #
  #
  # Enrique García

  u=0;
  v=1;
  z=2;
  phis=3;
  phin=4;

  cond=np.array([u,v,z,phin]).T;
  # cond=[u,v,z];
  i = 1
  ini = np.arange(cond[i-1],(ndivx+1)*ndivy*ndof+cond[i-1]+1,(ndivx+1)*ndof)
  bcl1=np.zeros((len(cond),len(ini)))
  for i in np.arange(1,len(cond)+1,1):
        bcl1[i-1,:]=np.arange(cond[i-1],(ndivx+1)*ndivy*ndof+cond[i-1]+1,(ndivx+1)*ndof);

  cond=np.array([u,v,z,phin]).T;
  i = 1
  ini = np.arange(ndivx*ndof+cond[i-1],((ndivx+1)*ndivy+ndivx)*ndof+cond[i-1]+1,(ndivx+1)*ndof)
  bcl3=np.zeros((len(cond),len(ini)))
  # cond=[u,v,z];
  for i in np.arange(1,len(cond)+1,1):
        bcl3[i-1,:]=np.arange(ndivx*ndof+cond[i-1],((ndivx+1)*ndivy+ndivx)*ndof+cond[i-1]+1,(ndivx+1)*ndof);


  cond=np.array([u,v,z,phis]).T;
  i = 1
  ini = np.arange(cond[i-1],ndivx*ndof+cond[i-1]+1,ndof)
  bcl2=np.zeros((len(cond),len(ini)))
  # cond=[u,v,z];
  for i in np.arange(1,len(cond)+1,1):
        bcl2[i-1,:]=np.arange(cond[i-1],ndivx*ndof+cond[i-1]+1,ndof);

  cond=np.array([u,v,z,phis]).T;
  i = 1
  ini = np.arange((ndivx+1)*ndivy*ndof+cond[i-1],((ndivx+1)*ndivy+ndivx)*ndof+cond[i-1]+1,ndof)
  bcl4=np.zeros((len(cond),len(ini)))
  # cond=[u,v,z];
  for i in np.arange(1,len(cond)+1,1):
        bcl4[i-1,:]=np.arange((ndivx+1)*ndivy*ndof+cond[i-1],((ndivx+1)*ndivy+ndivx)*ndof+cond[i-1]+1,ndof);

  bct=np.concatenate((bcl1,bcl2), axis=1)
  bct=np.concatenate((bct,bcl3), axis=1)
  bct=np.concatenate((bct,bcl4), axis=1)
  bct = bct.T
  [m,n]=np.shape(bct);
  bctt=bct[:,0]
  for i in np.arange(1,n,1):
        bctt=np.concatenate((bctt,bct[:,i]),axis=0);   

  bclong1 = np.arange(0,(ndivx+1)*(ndivy+1)*5+0.00001-1,5)
  bctt=np.concatenate((bctt,bclong1),axis=0); 
  bclong2 = np.arange(1,(ndivx+1)*(ndivy+1)*5+0.00001-1,5)
  bctt=np.concatenate((bctt,bclong2),axis=0); 
  bctt=np.sort(bctt).T
  bc = np.unique(bctt);


  return bc 


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


