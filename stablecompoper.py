import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.integrate import odeint,quad
from scipy.stats import kde,beta
import seaborn as sns
#%matplotlib
from importlib import reload
pi=np.pi
from scipy.optimize import brentq


from numpy import linalg as LA
from scipy.linalg import expm

from bdp import periodise


def partiefrac(x,T):
    y=x/T
    return(T*(y-int(y)))

def zchi(lam,mu,A,T,N):
    r"""genere N points du processus ponctuel forme des phases des differents individus dans un processus de naissance et de mort inhomogene
Si le processus s'éteint avant on a moins de N points"""
    z=[[0,0]] #initialisation : un individu au temps 0 (donc sa phase est 0)
    S=0 #S et linstant de saut courant du processus de Poisson deparametre A
    i=0 #i+1 doit etre l longueur de z
    while (len(z) <= N):
        echelle=len(z)*A
        S=S+rnd.exponential(scale=1/echelle)
        p=(lam(S) + mu(S))/A
        #print("S,p,z",S,p,z)
        u=rnd.uniform()
        if (u< p): #on accepte le temps de saut
            q=lam(S)/(lam(S)+mu(S))
            v=rnd.uniform()
            if (v<q):
                z.append([S,partiefrac(S,T)])
            else:
                j=rnd.choice(len(z))
                del z[j] #on enleve un individu au hasard
        if (len(z)==0):
            break #le processus s'est éteint
    return(z)

def estimdenszchi(lzero,muzero,T,N):
    def lam(t):
        return(lzero*(1+np.cos((2*pi*t)/T)))
    def mu(t):
        return(muzero)
    A=2*lzero+muzero
        
    z=np.array(zchi(lam,mu,A,T,N))
    if (len(z) != 0):
        w=z[:,1]
        k=kde.gaussian_kde(w)
        tt=np.linspace(0,T,100)
        plt.plot(tt,k(tt))
        plt.plot(tt,[1+np.cos((2*pi*t)/T) for t in tt],color="red")

#estimdenszchi(2,1,1,5000)  
    
def sestimdenszchi(lzero,muzero,T,N):
    def lam(t):
        return(lzero*(1+np.cos((2*pi*t)/T)))
    def primlam(t):
        r"une primitive de la fonction precedente"
        return (lzero*(t + (T/2*pi)*np.sin((2*pi*t)/T)))
    def mu(t):
        return(muzero)
    A=2*lzero+muzero
    tt=np.linspace(0,T,100)
    lamepa=np.array([lam(t)*np.exp(primlam(t))  for t in tt])
    lamepa=(lamepa/(lamepa.sum()))*100
    tlam=np.array([lam(t)  for t in tt])
    tlam=(tlam/tlam.sum())*100
    print("tlam.sum()",tlam.sum())
    while True:
        z=np.array(zchi(lam,mu,A,T,N))
        if (len(z) != 0):
            w=z[:,1]
            k=kde.gaussian_kde(w)
            plt.plot(tt,k(tt),label="estim par noyau echantillon")
            #plt.plot(tt,[(1+np.cos((2*pi*t)/T))/T for t in tt],color="red")
            plt.plot(tt,lamepa,color="green",label="densite")
            plt.plot(tt,tlam,color="red",label="$\lambda$")
            plt.legend()
            break


def nzchi(lam,mu,A,T,N):
    r"""genere des  points du processus ponctuel jusqu'à l'instant N T forme des phases des differents individus dans un processus de naissance et de mort inhomogene
Si le processus s'éteint avant on a moins de N points"""
    z=[[0,0]] #initialisation : un individu au temps 0 (donc sa phase est 0)
    S=0 #S et linstant de saut courant du processus de Poisson deparametre A
    i=0 #i+1 doit etre l longueur de z
    while (S < N*T):
        echelle=len(z)*A
        S=S+rnd.exponential(scale=1/echelle)
        p=(lam(S) + mu(S))/A
        #print("S,p,z",S,p,z)
        u=rnd.uniform()
        if (u< p): #on accepte le temps de saut
            q=lam(S)/(lam(S)+mu(S))
            v=rnd.uniform()
            if (v<q):
                z.append([S,partiefrac(S,T)])
            else:
                j=rnd.choice(len(z))
                del z[j] #on enleve un individu au hasard
        if (len(z)==0):
            break #le processus s'est éteint
    return(z)

def nsestimdenszchi(lzero,muzero,T,N,coeff=1.0,estimnoyau=False):
    r""" coeff est l'intensité de la modulation sinusoidale"""
    nbpts=100
    def lam(t):
        return(lzero*(1+coeff*np.cos(2*pi*t/T)))
    def primlam(t):
        r"une primitive de la fonction precedente"
        return (lzero*(t + coeff*(T/(2*pi))*np.sin((2*pi*t)/T)))
    def mu(t):
        return(muzero)
    A=(1+coeff)*lzero+muzero
    tt=np.linspace(0,T,nbpts)
    lamepa=np.array([lam(t)*np.exp(primlam(t))  for t in tt])
    lamepa=(lamepa/(lamepa.mean()))/T #normalisation
    tlam=np.array([lam(t)  for t in tt])
    tlam=(tlam/tlam.mean())/T #normalisation
    while True:
        z=np.array(nzchi(lam,mu,A,T,N))
        if (len(z) != 0):
            w=z[:,1]
            k=kde.gaussian_kde(w)
            print("tlam.sum()",tlam.sum(),"Taille de l'echantillon",len(w))
            plt.hist(w,density=True,label="histogram")
            if estimnoyau:
                plt.plot(tt,k(tt),label="echantillon")
            #plt.plot(tt,[(1+np.cos((2*pi*t)/T))/T for t in tt],color="red")
            plt.plot(tt,lamepa,color="green",label="stable composition density $\lambda(t) e^{A(t)}$ ")
            plt.plot(tt,tlam,color="red",label="$\lambda(t)$")
            plt.legend()
            plt.savefig("stablecompolbdsinusoid.pdf",bbox_inches='tight',dpi=150)
            break

#jeudi 23 avril : simplifions en prenant lambda consant et mu =0, T=1
def sisestimdenszchi(lzero,muzero,T,N):
    def lam(t):
        return(lzero)
    def primlam(t):
        r"une primitive de la fonction precedente"
        return (lzero*t)
    def mu(t):
        return(muzero)
    A=2*lzero+muzero
    tt=np.linspace(0,T,100)
    lamepa=np.array([lam(t)*np.exp(primlam(t))  for t in tt])
    lamepa=(lamepa/(lamepa.sum()))*100
    tlam=np.array([lam(t)  for t in tt])
    tlam=(tlam/tlam.sum())*100
    while True:
        z=np.array(nzchi(lam,mu,A,T,N))
        if (len(z) != 0):
            w=z[:,1]
            k=kde.gaussian_kde(w)
            print("tlam.sum()",tlam.sum(),"Taille de l'echantillon",len(w))
            plt.hist(w,density=True,label="histogramme")
            plt.plot(tt,k(tt),label="estim  noyau")
            #plt.plot(tt,[(1+np.cos((2*pi*t)/T))/T for t in tt],color="red")
            plt.plot(tt,lamepa,color="green",label="densite")
            plt.plot(tt,tlam,color="red",label="$\lambda$")
            
            plt.legend()
            break

#mardi 18 avril 2020
#un exemple de regime switching pour Sylvain
def swreg(t,xzero=[1,0],T=1,nbpts=50):
    r""" T est la periode, et t  le temps pendant lequel on fait tourner l'edo"""

    B=(2*np.log(2)/3)* np.array([[-2, 2], [1, -1]])
    BT=B.transpose()
    def msisi(x,s):
        y=s/T
        if (y-int(y)<0.5):
            M=B
        else:
            M=BT
        return(np.dot(M,x))
    def msisi1(x,s):
        return(np.dot(B,x))
    def msisi2(x,s):
        return(np.dot(BT,x))
     
    timeint=np.linspace(0,t,1+int(t/T)*nbpts)
    z=np.array((odeint(msisi,xzero,timeint)))
    z1=np.array((odeint(msisi1,xzero,timeint)))
    z2=np.array((odeint(msisi2,xzero,timeint)))
    plt.plot(timeint,z[:,0],label=" H switching")
    plt.plot(timeint,z[:,1],label=" V switching")
    #plt.plot(timeint,z1,label="fixe 1")
    plt.plot(timeint,z2[:,0],label="H fixed")
    plt.plot(timeint,z2[:,1],label="V fixed")
    plt.legend()
    plt.savefig("switchingvsfixed.pdf",bbox_inches='tight',dpi=150)
    
 
