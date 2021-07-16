# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 17:02:45 2021

@author: Enrique GM
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from numpy.linalg import eig
from scipy.interpolate import interp1d
import numpy.matlib
from statsmodels.distributions.empirical_distribution import ECDF


def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days = 366)
    return day + dayfrac


# Read results

filename="Resonant_frequencies.txt"
expfreq = np.loadtxt(filename)

filename="Dates.txt"
Dates = np.loadtxt(filename)
Datespy = [matlab2datetime(tval) for tval in Dates]


plt.figure(1)
ax = plt.gca()
for i in np.arange(0,4+1,1):
   plt.scatter(Datespy,expfreq[:,i], s=2)
formatter = mdates.DateFormatter("%d-%m-%Y")
ax.xaxis.set_major_formatter(formatter)
plt.ylabel('Resonant frequency [Hz]')

# Fill missing data

for i in np.arange(0,4+1,1):
     poscero = np.where(expfreq[:,i] == 0)[0]
     expfreq[poscero,i] = np.nan  
     x = np.arange(1,np.size(expfreq[:,i])+1,1)
     bi = np.isnan(expfreq[:,i])
     comp = np.empty_like(expfreq)
     f = interp1d(x[~bi], expfreq[~bi,i],kind='nearest', fill_value="extrapolate")
     expfreq[bi,i]= f(x[bi])

plt.figure(2)
ax = plt.gca()
for i in np.arange(0,4+1,1):
   plt.scatter(Datespy,expfreq[:,i], s=2)
formatter = mdates.DateFormatter("%d-%m-%Y")
ax.xaxis.set_major_formatter(formatter)
plt.ylabel('Resonant frequency [Hz]')

# We set_up a training period

tp = 365*24*2   # One year

trainingpop = expfreq[0:tp,:]


# Step 1 - Create statistical model


# 1.1. Normalization
x = trainingpop
meanval = np.mean(x,axis=0);
stadval = np.std(x,axis=0);
x = (x-np.matlib.repmat(meanval,np.size(x,0),1))/np.matlib.repmat(stadval,np.size(x,0),1);


# 1.2. PCA dimension reduction

COV = np.cov(x.T)
D, V = eig(COV)

sort_index = np.flip(np.argsort(D))

latent2 = D[sort_index];
coeff2 = V[:,sort_index];
D = latent2;
explainedvar = 100*D/np.sum(D);
score2 = np.matmul(x,coeff2)

plt.figure(3)
plt.subplot(1,2,1)
plt.bar([1,2,3,4,5],explainedvar)
plt.xticks([1,2,3,4,5], ('PC1', 'PC2', 'PC3', 'PC4', 'PC5'))
plt.ylabel('Explained variance [%]')
plt.subplot(1,2,2)
plt.bar([1,2,3,4,5],np.cumsum(explainedvar))
plt.xticks([1,2,3,4,5], ('PC1', 'PC2', 'PC3', 'PC4', 'PC5'))
plt.ylabel('Cumulated explained variance [%]')


# 1.3. Dimension reduction

ll = 2
Z_vect=np.matmul(x,coeff2)
score2=Z_vect[:,np.arange(0,ll,1)];
reconstr=np.matmul(score2,coeff2[:,np.arange(0,ll,1)].T)
reconstr = reconstr*np.matlib.repmat(stadval,np.size(x,0),1)+np.matlib.repmat(meanval,np.size(x,0),1);


plt.figure(4)
ax = plt.gca()
for i in np.arange(0,4+1,1):
   plt.scatter(Datespy[1:tp],trainingpop[1:tp,i], s=2,c='b')
   plt.scatter(Datespy[1:tp],reconstr[1:tp,i], s=2,c='r')
formatter = mdates.DateFormatter("%d-%m-%Y")
ax.xaxis.set_major_formatter(formatter)
plt.ylabel('Resonant frequency [Hz]')

# 1.4 Residuals

R = reconstr-trainingpop;

plt.figure(5)
for i in np.arange(0,4+1,1):
   axx = plt.subplot(5,2,2*(i+1)-1);
   plt.scatter(Datespy[0:tp],R[:,i], s=2, c='b')
   formatter = mdates.DateFormatter(" ")
   if i==6:
       formatter = mdates.DateFormatter("%d-%m-%Y")
   axx.xaxis.set_major_formatter(formatter)
   plt.ylabel('R = '+str(i+1))

for i in np.arange(0,4+1,1):
   axx = plt.subplot(5,2,2*(i+1))
   plt.hist(R[:,i],bins=40)
   plt.xlim([-0.4,0.3])
   if i<4:
       axx.get_xaxis().set_ticks([])
       

# Now we use the statistical model to perform predictions

x = expfreq;
meanval = np.mean(x,axis=0);
stadval = np.std(x,axis=0);
x = (x-np.matlib.repmat(meanval,np.size(x,0),1))/np.matlib.repmat(stadval,np.size(x,0),1);

Z_vect=np.matmul(x,coeff2)
score2=Z_vect[:,np.arange(0,ll,1)]
reconstr=np.matmul(score2,coeff2[:,np.arange(0,ll,1)].T)
reconstr = reconstr*np.matlib.repmat(stadval,np.size(x,0),1)+np.matlib.repmat(meanval,np.size(x,0),1);


plt.figure(6)
ax = plt.gca()
for i in np.arange(0,4+1,1):
   plt.scatter(Datespy,expfreq[:,i], s=2,c='b')
   plt.scatter(Datespy,reconstr[:,i], s=2,c='r')
formatter = mdates.DateFormatter("%d-%m-%Y")
plt.plot([Datespy[tp],Datespy[tp]],[2,8], '--k')
ax.xaxis.set_major_formatter(formatter)
plt.ylabel('Resonant frequency [Hz]')
plt.ylim([2,8])


# Hotelling's control chart

# Residuals
R = reconstr-expfreq;

# Parameters
gr = 4 # Group number
UCL_lim = 0.95

# Phase I
cov_mat_data=np.cov(R[0:tp,:].T)
inv_cov_mat_data = np.linalg.pinv(cov_mat_data)
mean_v=np.mean(R[0:tp,:],axis=0)
              
# Phase II
s=int(np.floor(tp/gr))
t_quadro=0;


p = np.size(R,1); # Number of variables
n = gr; # Group size
m = np.size(R,0)/gr; # Number of observations

# T2-statistic
t_quadro = np.zeros((int(m),1))
for jj in np.arange(1,m,1,dtype=int):
     dif = np.mean(R[(jj-1)*gr:jj*gr,:],axis=0) - mean_v
     t_quadro[jj-1,0] = n*np.matmul(np.matmul(dif,inv_cov_mat_data),dif.T)

 
# Control chart
n_cl=tp
xx=np.linspace(0,np.max(t_quadro[0:s,0]), endpoint=True)
        
t_quadro_new=t_quadro[0:s]  # Only the part corresponding to the training period

# Upper Control Limit (UCL)
ecdf = ECDF(t_quadro_new[:,0],side='right')
F = ecdf.y
X = ecdf.x

pos=np.argmin(np.abs(F-UCL_lim)/UCL_lim);
ucl=X[pos]

time_new = []
for kk in np.arange(1,np.size(t_quadro)+1,1):
    time_new.append(Datespy[kk*gr])

time_new = np.array(time_new)

plt.figure(7)
ax = plt.gca()
pos = t_quadro>=ucl
neg = t_quadro<ucl
neg = np.where(neg)[0]
pos = np.where(pos)[0]
plt.scatter(time_new[pos[:]],t_quadro[pos],c='r',s=2)
plt.scatter(time_new[neg[:]],t_quadro[neg],c='g',s=2)
plt.plot([time_new[0],time_new[-1]],[ucl,ucl],'--b', lineWidth=2)
plt.plot([Datespy[tp],Datespy[tp]],[0,200],'--r', lineWidth=2)
formatter = mdates.DateFormatter("%d-%m-%Y")
plt.plot([Datespy[tp],Datespy[tp]],[2,8], '--k')
ax.xaxis.set_major_formatter(formatter)
plt.ylabel('Hotellings T2')
plt.ylim([0,200])
plt.xlim([time_new[0],time_new[-1]])
plt.ylim([0,200])
