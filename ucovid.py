import numpy as np
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

#pour matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)
#plt.rc('text', usetex=True)
plt.rc('xtick',labelsize=22)
plt.rc('ytick',labelsize=22)

#mardi 31 mars 2020
#essayons tout d'abord d'écrire des fonctions qui calculent le rayon spectral
#et l'abcisse de convergence d'une matrice

from numpy import linalg as LA
from scipy.linalg import expm

def spectralabc(m):
    """m is a matrix"""
    return(LA.eigvals(m).real.max())

def spectralrad(M):
    """M is a matrix : returns the spectral radius"""
    return(np.absolute(LA.eigvals(M)).max())

#et on teste

A=np.array([[1, -1], [4, 2]])
B=np.diag((1, 2, 3))
          
ei=LA.eigvals(A)
z=ei[0]
rei=ei.real

np.exp(spectralabc(A))
spectralrad(expm(A)) #doit donner la même chose

#un premier modele de covid avec deux classes Asymptomatique et Infectieux
def tauxcontacper(beta,p,cbeta,T):
    """renvoie une fonction de contact de periode T qui vaut beta pendant une fraction p de laperiode et beta(1-cbeta) pendant le reste de la periode"""
    def f(t):
        if (t <= T*p):
            return(beta)
        else:
            return(beta*(1-cbeta))
    return(f)
def periodise(f,T=1):
    #retourne la fonction qui etait definie sur [0,T] periodisee sur R
    def g(t):
        return(f(t-T*np.floor(t/T)))
    return(g)
T=7
p=0.3
tt=np.linspace(0,T,100)
f=tauxcontacper(0.25,p,0.8,T)
#plt.plot(tt,[f(s) for s in tt])

dtt=np.linspace(-2*T,3*T,400)
g=periodise(f,T)
#plt.plot(dtt,[g(s) for s in dtt])

def lamat(betaA,betaS,piS,gammaA,gammaS):
    return np.array([[piS*betaS-gammaS,piS*betaS],[(1-piS)*betaA,(1-piS)*betaA-gammaA]])

def lesabcissesspec(betaA,betaS,piS,gammaA,gammaS,cbeta):
    azero=lamat(betaA,betaS,piS,gammaA,gammaS)
    azcbeta=lamat(betaA*(1-cbeta),betaS*(1-cbeta),piS,gammaA,gammaS)
    return(spectralabc(azero),spectralabc(azcbeta))

def matcroissance(betaa,betai,pii,gammai,gammaa):
    def a(t):
        return np.array([[pii*betai(t) -gammai,pii*betaa(t)],
                         [(1-pii)*betai(t),(1-pii)*betaa(t)-gammaa]])

    return(a)
betaamax=0.25
betaimax=0.25
cbeta=0.8
pii=0.15
p=0.3
gammaa=0.1
gammai=0.05
betaa=tauxcontacper(betaamax,p,cbeta,T)
betai=tauxcontacper(betaimax,p,cbeta,T)
#plt.plot(tt,[betaa(s) for s in tt])
a=matcroissance(betaa,betai,pii,gammai,gammaa)
spectralabc(a(1)),spectralabc(a(5))
#puis la on calcule la composee des exponentielles de matrices
phiT=np.dot(expm(a(5)*(1-p)),expm(a(1)*p))
np.log(spectralrad(phiT)),p*spectralabc(a(1))+(1-p)*spectralabc(a(5))
#l'approximation du rayonspectral par l'integrale de l'abcisse spectrale
#n'est pas si mauvaise que cela.

#verifions que si gammai=gammaa, alors il n'y a qu'une classe d'infecte, et le rzero c'est beta/gamma

b=matcroissance(betaa,betaa,pii,gammaa,gammaa)
spectralabc(b(1)),spectralabc(b(5)) #on obtient les beta -gamma pour les deux périodes de temps
phiT=np.dot(expm(b(5)*(1-p)),expm(b(1)*p))
np.log(spectralrad(phiT)),p*spectralabc(b(1))+(1-p)*spectralabc(b(5))

#tracons la courbe de Uri Alon
sns.set(style="whitegrid")
def ualon(cbeta,rzero=2.5):
    return( (1-rzero*(1-cbeta))/(rzero*cbeta))
rzero=2.5
utt=np.linspace(1-1/rzero,1,100)
#plt.xlabel(r"$c_\beta$ : efficiency of social distancing")
#plt.ylabel("p : proportion of freedom (no  social distancing)")
#plt.plot(utt,[ualon(i,rzero) for i in utt])

#mercredi premier avril 2020
#tracons le rayon spectral pour une periode en fonction de p, avec cbeta donne

def lrsp(p,T=1):
    betaa=tauxcontacper(betaamax,p,cbeta,T)
    betai=tauxcontacper(betaimax,p,cbeta,T)
    #plt.plot(tt,[betaa(s) for s in tt])
    a=matcroissance(betaa,betai,pii,gammai,gammaa)
    phiT=np.dot(expm(a(0.01*T)*p*T),expm(a(0.99*T)*(1-p)*T))
    return((np.log(spectralrad(phiT)))/T)

#ptt=np.linspace(0,1,100)
#plt.plot(ptt,[lrsp(p,1) for p in ptt])

#on voit que cela ne depend presque pas de la periode
#plt.plot(ptt,[lrsp(p,7) for p in ptt])

#lancons maintenant la recherche du point d'annulation
brentq(lambda a: lrsp(a,T=7),0,1)

#puis faisons le trace de la courbe p fonction de cbeta
def siraipcbeta(T=1,nbpts=50):
    ctt=np.linspace(0,1,nbpts)
    l=[]
    for cbeta in ctt:
        def lrsp(p):
            betaa=tauxcontacper(betaamax,p,cbeta,T)
            betai=tauxcontacper(betaimax,p,cbeta,T)
            a=matcroissance(betaa,betai,pii,gammai,gammaa)
            phiT=np.dot(expm(a(0.01*T)*p*T),expm(a(0.99*T)*(1-p)*T))
            return((np.log(spectralrad(phiT)))/T)
        if (lrsp(0)*lrsp(1)<0):
            p=brentq(lrsp,0,1)
            l.append([cbeta,p])
    return(l)

# l=np.array(siraipcbeta(T=7))

# f,ax=plt.subplots(2,1)
# axc=ax[0]
# axc.set_xlabel(r"$c_\beta$ : efficiency of social distancing")
# axc.set_ylabel("p : proportion of freedom (no  social distancing)")
# axc.plot(utt,[ualon(i,rzero) for i in utt])
# axc.plot(l[:,0],l[:,1])
# axc=ax[1]
# axc.plot(l[:,0],l[:,1])


#ecrivns une fonction que nous rendrons interactive
def siraicov(betaA=0.25,
             betaS=0.25,
             piS=0.15,gammaA=0.1,gammaS=0.05,T=7,nbpts=50):
    
    ctt=np.linspace(0,1,nbpts)
    l=[]
    for cbeta in ctt:
        def lrsp(p):
            fbetaA=tauxcontacper(betaA,p,cbeta,T)
            fbetaS=tauxcontacper(betaS,p,cbeta,T)
            a=matcroissance(fbetaA,fbetaS,piS,gammaS,gammaA)
            phiT=np.dot(expm(a(0.99*T)*(1-p)*T),expm(a(0.01*T)*p*T))
            return((np.log(spectralrad(phiT)))/T)
        if (lrsp(0)*lrsp(1)<0):
            p=brentq(lrsp,0,1)
            l.append([cbeta,p])
    l=np.array(l)
    
    f,ax=plt.subplots(1,1)
    axc=ax
    axc.set_xlabel(r"$c_\beta$ : efficiency of social distancing")
    axc.set_ylabel("p : proportion of freedom (no  social distancing)")
    axc.plot(utt,[ualon(i,rzero) for i in utt])
    axc.plot(l[:,0],l[:,1])



def bsiraicov(betaA=0.25,
             betaS=0.25,
             piS=0.15,gammaA=0.1,gammaS=0.05,T=7,nbpts=50):
    
    ctt=np.linspace(0,1,nbpts)
    l=[]
    la=[]
    for cbeta in ctt:
        def lrsp(p):
            fbetaA=tauxcontacper(betaA,p,cbeta,T)
            fbetaS=tauxcontacper(betaS,p,cbeta,T)
            a=matcroissance(fbetaA,fbetaS,piS,gammaS,gammaA)
            phiT=np.dot(expm(a(0.99*T)*(1-p)*T),expm(a(0.01*T)*p*T))
            return((np.log(spectralrad(phiT)))/T)
        if (lrsp(0)*lrsp(1)<0):
            p=brentq(lrsp,0,1)
            l.append([cbeta,p])
        saz,sazcb=lesabcissesspec(betaA,betaS,piS,gammaA,gammaS,cbeta)
        #print("saz,sazcb",saz,sazcb)
        if (sazcb<0.0):
            #print("\t :saz,sazcb",saz,sazcb)
            la.append([cbeta,sazcb/(sazcb-saz)])
    l=np.array(l)
    la=np.array(la)
    #print("l-la",l-la)
    f,ax=plt.subplots(1,1)
    axc=ax
    axc.set_xlabel(r"$c_\beta$ : efficiency of social distancing")
    axc.set_ylabel("p : proportion of freedom (no  social distancing)")
    axc.plot(utt,[ualon(i,rzero) for i in utt],label="Ualon")
    axc.plot(l[:,0],l[:,1],label="true critical line")
    axc.plot(la[:,0],la[:,1],label="approximate critical line")
    axc.legend(loc='upper left')
    axc.set_title("T="+str(T))


#jeudi 2 avril 2020 : il faut que je verifie mon theoreme sur les abcisses spectrales
A=lamat(betaamax,betaimax,pii,gammaa,gammai)
B=lamat(betaamax*(1-cbeta),betaimax*(1-cbeta),pii,gammaa,gammai)
[np.log(spectralrad(np.dot(expm(B*(1-p)*T),expm(A*p*T))))/T for T in 10*np.arange(1,40)]
spectralabc(A)*p + spectralabc(B)*(1-p)#pas la meme quantite
spectralabc(A)-np.log(spectralrad(expm(A)))#la cela coincide
#il faut prendre T del 'orde de 400 pour que cela se rapproche!!!                     


#on trace maintenant avec deux périodes pour en voir l'influence
def bipersiraicov(betaA=0.25,
             betaS=0.25,
                  piS=0.15,gammaA=0.1,gammaS=0.05,T1=7,T2=100,nbpts=50):
    
    ctt=np.linspace(0,1,nbpts)
    l=[[],[]]
    for i, T in enumerate((T1,T2)):
        for cbeta in ctt:
            def lrsp(p):
                fbetaA=tauxcontacper(betaA,p,cbeta,T)
                fbetaS=tauxcontacper(betaS,p,cbeta,T)
                a=matcroissance(fbetaA,fbetaS,piS,gammaS,gammaA)
                phiT=np.dot(expm(a(0.99*T)*(1-p)*T),expm(a(0.01*T)*p*T))
                return((np.log(spectralrad(phiT)))/T)
            if (lrsp(0)*lrsp(1)<0):
                p=brentq(lrsp,0,1)
                l[i].append([cbeta,p])
    l=np.array(l)
    
    f,ax=plt.subplots(1,1)
    axc=ax
    axc.set_xlabel(r"$c_\beta$ : efficiency of social distancing")
    axc.set_ylabel("p : proportion of freedom (no  social distancing)")
    axc.plot(utt,[ualon(i,rzero) for i in utt],label="U Alon")
    axc.plot(l[0][:,0],l[0][:,1],label="T="+str(T1))
    axc.plot(l[1][:,0],l[1][:,1],label="T="+str(T2))
    axc.legend(loc='upper left')
    axc.set_title(r"critical curves : $p(c_\beta)$")

