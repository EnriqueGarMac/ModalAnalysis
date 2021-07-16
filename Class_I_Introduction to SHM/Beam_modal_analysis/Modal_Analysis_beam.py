# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 20:03:24 2021

@author: Enrique GM
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Example of modal analysis of simply supported beam

L=8
EI=30*10**9*(1./12)*0.15*0.2**3
rho=2500*0.15*0.2
chi=2.0/100


nmodos=4
x = np.linspace(0,L,55,endpoint=True)
freq = np.zeros((nmodos,1))
PHI = np.zeros((len(x),nmodos))
for j in np.arange(0,nmodos,1):
    for xx in np.arange(0,len(x),1):
      n=j+1;
      wn=n**2*np.pi**2*np.sqrt(EI/(rho*L**4));
      wd=wn*np.sqrt(1-chi**2);
      freq[j]=wn/(2*np.pi);
      PHI[xx,j]=np.sin(n*np.pi*x[xx]/L);



plt.figure(2)
plt.subplot(2,2,1)
plt.plot(x,PHI[:,0],Linewidth=2)
plt.plot(x,PHI[:,0]*0,'--k',Linewidth=2)
plt.ylim([-1,1])
plt.xlabel('x')
plt.title('Freq.: '+"{:.2f}".format(float(freq[0,:]))+ 'Hz')
plt.subplot(2,2,2)
plt.plot(x,PHI[:,1],Linewidth=2)
plt.plot(x,PHI[:,0]*0,'--k',Linewidth=2)
plt.ylim([-1,1])
plt.xlabel('x')
plt.title('Freq.: '+"{:.2f}".format(float(freq[1,:]))+ 'Hz')
plt.subplot(2,2,3)
plt.plot(x,PHI[:,2],Linewidth=2)
plt.plot(x,PHI[:,0]*0,'--k',Linewidth=2)
plt.ylim([-1,1])
plt.xlabel('x')
plt.title('Freq.: '+"{:.2f}".format(float(freq[2,:]))+ 'Hz')
plt.subplot(2,2,4)
plt.plot(x,PHI[:,3],Linewidth=2)
plt.plot(x,PHI[:,0]*0,'--k',Linewidth=2)
plt.ylim([-1,1])
plt.xlabel('x')
plt.title('Freq.: '+"{:.2f}".format(float(freq[3,:]))+ 'Hz')
plt.plot(x,PHI[:,0]*0,'--k',Linewidth=2)
plt.tight_layout()


# Mode shape animation
mplot = 4

x_data = x
y_data = PHI[:,mplot-1]

fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(-1, 1)
line, = ax.plot(x_data, y_data)

def animation_frame(i):
	line.set_ydata(y_data*np.cos(i*2*np.pi/15))
	return line, 

animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, 15, 00.1), interval=10)
plt.show()
animation.save('mode_shape.gif', writer='imagemagick', fps=30)