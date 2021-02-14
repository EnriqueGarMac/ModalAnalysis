# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 18:04:10 2021

@author: Enrique GM
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats


N = 100000
stdv = 1
meanv = 2


# Plot analytical distribution
plt.figure(1)
plt.subplot(2,1,1)
x = np.linspace(-12,12,1000);
y = norm.pdf(x,meanv,stdv);
plt.plot(x,y,linewidth=2)
plt.ylabel('pdf')
plt.subplot(2,1,2)
x = np.linspace(-12,12,1000)
y = norm.cdf(x,meanv,stdv)
plt.plot(x,y,linewidth=2)
plt.ylabel('cdf')
plt.tight_layout()
plt.savefig('pdf_cdf.jpg',dpi=300)

plt.figure(2)
N = 1000
X = stdv*(np.random.randn(N,1)+meanv);
plt.plot(X)
plt.savefig('random_signal.jpg',dpi=300)

# Mean value
mean_X = (1/N)*np.sum(X)
print('Mean value: ')
print(mean_X)
print(np.mean(X))

# Variance
var = (1/(N-1))*np.sum((X-mean_X)**2);
stdvar = np.sqrt(var)
print('Standard deviation: ')
print(stdvar)
print(np.std(X, ddof=1))

# Skewness
skewnessval = (1/(N))*np.sum((X-mean_X)**3)/(np.sqrt((1/(N))*np.sum((X-mean_X)**2)))**3
print('Skewness: ')
print(skewnessval)
print(float(stats.skew(X)))

# Kurtosis
Kurtosis =  (1/(N))*np.sum((X-mean_X)**4)/(np.sqrt((1/(N))*np.sum((X-mean_X)**2)))**4
print('Skewness: ')
print(Kurtosis)
print(float(stats.kurtosis(X, fisher = False)))


# HISTOGRAM AND CUMMULATIVE COUNTS

plt.figure(3)
plt.subplot(1,2,1)
hist, bin_edges = np.histogram(X,bins=24);
plt.bar(0.5*(bin_edges[0:-1]+bin_edges[1:]),hist)
plt.ylabel('Sample pdf')
plt.subplot(1,2,2)
plt.bar(0.5*(bin_edges[0:-1]+bin_edges[1:]),np.cumsum(hist))
plt.ylabel('Sample cdf')
plt.tight_layout()
plt.savefig('pdf_cdf_hist.jpg',dpi=300)