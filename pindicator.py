import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.integrate import odeint,quad
from scipy.stats import kde,beta
import seaborn as sns
from importlib import reload
pi=np.pi



import ucovid

####### Calcul de l'indicateur P de Heesterbek et Roberts
#########"" 28 septembre 2020

#la donnee d'une matrice 1 periodique est faite via la fonction gena

def pindic(gena,T,nbpts=100,voir=False):
    def a(t):
        return(gena(t/T))
    def msisi(x,t):
        return(np.dot(a(t),x))
    

    #on determine la solution fondamentale
    timeint=np.arange(0,T+1/nbpts,T/nbpts)
    dim=(a(0).shape)[0]
    z=np.zeros(shape=(nbpts+1,dim,dim))
    for i in range(dim):
        y=np.zeros(shape=dim)
        y[i]=1.0
        z[:,:,i]=np.array(odeint(msisi,y,timeint))
    #la matrice de monodromie est obtenue en prenant pour colonnes les valeurs
    #des solutions au temps T
    E=np.array([z[-1,:,i] for i in range(dim)])
    E=E.transpose()
    l,v=ucovid.vecetspectralrad(E)

    #puis on considere la solution x issue du vecteur propre de la matrice de
    #monodromie
    x=v[0]*z[:,:,0]
    for i in np.arange(1,dim):
        x=x+v[i]*z[:,:,i]
    #supposons que les elements diagonaux de A soient constants negatifs
    #histoire de simplifier les calculs
    P=1.0
    azero=a(0)
    if voir:
        print("azero=",azero)
        #la correction pour obtenir une solution periodique
        corr=np.exp(-timeint*np.log(l))
    for i in range(dim):
        xi=x[:,i]
        ip1=(i+1)% dim
        xm=np.array([(a(timeint[j])[ip1,i])*xi[j] for j in range(len(xi))])
        if voir:
            plt.plot(xi*corr,label="x["+str(i))
            plt.plot(xm,label="xm["+str(i))
            print("P=",P,"sommes : xi=",np.sum(xi),"xmi=",np.sum(xm),-azero[ip1,ip1])
        P=P*(np.sum(xm))
        P=P/(np.sum(xi))
        P=P/(-azero[ip1,ip1])
    if voir:
        plt.plot([a(timeint[j])[1,0] for j in range(len(xi))],label="a(t)")
        plt.legend()
        return(l,P)
    
    return(P)




def genAH(t):
    return(ucovid.genafricanhorseper(epsilon=0.5,t=t))

azero=genAH(0)

#pindic(genAH,T=1)

def genazero(t):
    return(np.array([[-2,2],[1,-1]]))


#puis maintenant on regarde la dependance de P en espilon
def pepsindic(epsmax,genepsa,T,nbeps=50,voir=False):
    ept=np.linspace(0.0,epsmax,nbeps)
    pe=np.zeros(shape=nbeps)
    for i,e in enumerate(ept):
        def gena(t):
            return(genepsa(e,t))
        pe[i]=pindic(gena,T=T)
    if voir:
        plt.plot(ept,np.log(pe))
    return(pe)

#pour le modele africanhorse
#pepsindic(0.5,ucovid.genafricanhorseper,T=1,voir=True)

#pour le modele bacaer
def genbaca(e,t):
    return(ucovid.genex2per(epsilon=e,t=t,b12=0,b21=1))

#pepsindic(0.5,genbaca,T=1,voir=True)

def genbaca1(t):
    return(genbaca(1,t))
def genbaca0(t):
    return(genbaca(0,t))


#maintenant tracons les trois courbes
#(1/T) ln(lambda_d), msa et ln(P) en fonction de epsilon

def plamsa(gena,epsilonmax=0.5,T=1,recalage=True):
    nbeps=50
    ept=np.linspace(0.0,epsilonmax,nbeps)
    x=np.array([ucovid.lamsaetapp(gena,epsilon=e,T=T) for e in ept])
    lamd=x[:,0]
    ustlnlam=np.log(lamd)/T
    if recalage:
        ustlnlam=ustlnlam-ustlnlam[0]
    plt.plot(ept,ustlnlam,label=r"$\frac{1}{T} \ln(\lambda_d)$") #on voit bien que c'est en delta^2
    msa=x[:,1]
    if recalage:
        msa = msa -msa[0]
    plt.plot(ept,msa,label=r"$MSA=\int s(A(u))\, du$")
    pe=np.zeros(shape=nbeps)

    for i,e in enumerate(ept):
        def mongena(t):
            return(gena(e,t))
        pe[i]=pindic(mongena,T=T)
    print("P[0]=",pe[0])
    pe=pe/pe[0]
    plt.plot(ept,np.log(pe),label="ln(P/P[0])")
    plt.xlabel(r"$\epsilon$")
    plt.legend()
    #plt.savefig("ex2msaperiodic.pdf",bbox_inches='tight' )


#maintenant il faut voir si l'indicateur P resite au decalage du rayon spectral
#s'il varie dans le bon sens
def genbacadecal(e,t,dec=-0.5):
    return(genbaca(e,t)+ dec*np.identity(2))

