# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:56:13 2015
The Huber_braun neuronal model function
@author: ksxuu
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
#import qr_decomposition

def Lor(X,t):
    x,y,z=X
    return np.array([s*(y-x),
            r*x-y-x*z,
            x*y-b*z])

def JacLor(t,X):
    x,y,z=X
    Jac=np.array([[-s, s, 0],
                  [r-z, -1, -x],
                  [y, x, -b]])
    return Jac

"""
The main function is starting from here
"""

#Parámetros de este Modelo
s,b,r = (10,8/3,28)

#initial value
x=1; y=0; z=0

# ADAPTATION SIMULATION
#initial value
X=np.array([x,y,z])
adaptTime=200
adaptInt=0.01  #This controls only for the returned values, not the calculation
Tadapt=np.arange(0,adaptTime,adaptInt)
Yadapt=integrate.odeint(Lor,X,Tadapt)

# SIMULATION FOR LEs CALCULATION
X0=Yadapt[-1]  #initial conditions after some adaptation

tBegin=0
tEnd=80
deltat=0.002   #This is time interval between Jacobian calculations
T= np.arange(tBegin, tEnd, deltat)

Q=np.identity(len(X0))*0.1
X=X0
LCE_T=[]
norm_T=[]

for i in T:
    X=integrate.odeint(Lor,X,(0,deltat))[-1]
    B=Q + deltat*np.dot(JacLor(i,X),Q)
    Q,R=np.linalg.qr(B)
#    Q,R=qr_decomposition.gram_schmidt_process(B)
    LCE_T.append(np.log(np.abs(np.diag(R)))/deltat)

#%%
i_init=10

LCE_T=np.array(LCE_T)
LCEv=np.cumsum(LCE_T[i_init:,:],0)/(T/deltat+deltat)[i_init:,None]

print(LCEv[-1])

plt.figure(1,figsize=(10,4))
plt.clf()

plt.subplot(121)
plt.plot(T[i_init:],LCEv[:,:2])
plt.xlabel("Time")
plt.ylabel("LCE average")

plt.subplot(122)
plt.plot(T[i_init:],LCEv)
plt.xlabel("Time")

plt.figure(2,figsize=(10,4))
plt.clf()
plt.subplot2grid((1,3),(0,0),colspan=2)
plt.plot(Tadapt,Yadapt)
plt.xlabel("Time")
plt.legend(("x","y","z"))

plt.subplot2grid((1,3),(0,2),projection='3d')
plt.plot(Yadapt[2000:,0],Yadapt[2000:,1],Yadapt[2000:,2])

