import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.integrate import odeint,quad
from scipy.stats import kde,beta
#import seaborn
#import mpmath
#from mpmath import mp
#from numba import jit
from importlib import reload
pi=np.pi
from functools import reduce
from scipy.interpolate import CubicSpline
from matplotlib.patches import Rectangle
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)
#plt.rc('font', family='serif',size='16')
plt.rc('text', usetex=True)
plt.rc('xtick',labelsize=22)
plt.rc('ytick',labelsize=22)
#plt.rc('axes', labelsize=22)
#plt.rc('axes', titlesize=22) 
#couleurcontrole='cyan'
couleurcontrole='#00AFF0'
cgris='lightgrey'


def bandegris(a,b,couleur=cgris,alpha=1.0):
    plt.axvspan(a, b, color=couleur, alpha=alpha, lw=0)
def axbandegris(ax,a,b,couleur=cgris,alpha=1.0):
    ax.axvspan(a, b, color=couleur, alpha=alpha, lw=0)


#pour trouver le max d'une fonction à 2 paramètres 
#@jit
def tm(f,T,N):
    N=int(N)
    x=0
    y=1/N
    m=f(x,y)
    for i in range(N):
        for j in range(i+1,N+1):
            z=f(i/N,j/N)
            if (z>m):
                x=i/N
                y=j/N
                m=z
    return((x,y,m))

#tm(lambda x,y : y*y-x*x,1.0,5000)

def rzero(la,mu,T):
    lb=quad(la,0,T)[0]
    mb=quad(mu,0,T)[0]
    return(lb/mb)
  

def lam(x):
    return(1.5*(1 +np.sin(2*pi*x)))
def mu(x):
    return(1.0)

rzero(lam,mu,1)


def optimrzero(la,mu,rhom,T,N=100):
    lb=quad(la,0,T)[0]
    mb=quad(mu,0,T)[0]
    K= (lb-mb)/rhom
    print("K=",K)
    m=0.0
    tun,tdeux=0,0
    c=T
    h=T/N #le pas
    for i in range(N):
        for j in range(i+1,N+1):
            z=quad(la,i*h,j*h)[0]
            if (z>K):
                ci=(j-i)*h
                print("i,j,ci",i,j,ci)
                if(ci < c):
                    tun,tdeux=i*h,j*h
                    c=ci
                    print("tun,tdeux,ci",tun,tdeux,ci)
                break
    return(tun,tdeux,c)
    

#optimrzero(lam,mu,T=1.0,rhom=0.5)



#la fonction psi
def psi(al):
    return(al + (1/pi)*np.sin(pi*al))



#allons y pour optimiser la proba d'émergence


#mardi 27 novembre. Exemple du creneau. Comparaison de deux stratégies.

def scaling(f,T):
    def g(t):
        return(f(t/T))
    return(g)
def Scaling(f,T):
    def g(t):
        return (T*f(t/T))
    return(g)
    
def calculepe(lap,Lap,mup,Mup,T):
    n=1- np.exp(-T*(Lap(1)-Mup(1)))
    def f(t):
        a=quad(lambda x : lap(x)*np.exp(-T*(Lap(x)-Mup(x))),0,t)[0]
        b=quad(lambda x : lap(x)*np.exp(-T*(Lap(x)-Mup(x))),t,1)[0]
        d=np.exp(T*(Lap(t) -Mup(t)))*(b +(1-n)*a)*T
        return (n/d)
    return(f) #on retourne une fonction t -> pe(tT,T) pour t dans (0,1)
def newcalculepe(lap,Lap,mup,Mup,T,limit=50):
    phiun=Lap(1)-Mup(1)
    n=1- np.exp(-T*phiun)
    def integrand(x):
        if (x<1):
            return(lap(x)*np.exp(-T*(Lap(x)-Mup(x))))
        else:
            return(lap(x-1)*np.exp(-T*(Lap(x-1)-Mup(x-1)+phiun)))
    def f(t):
        a=quad(integrand,t,t+1,limit=limit)[0]
        d=np.exp(T*(Lap(t) -Mup(t)))*a*T
        return (n/d)
    return(f) #on retourne une fonction t -> pe(tT,T) pour t dans (0,1)

def nncalculepe(lap,Lap,mup,Mup,T,limit=50):
    phiun=Lap(1)-Mup(1)
    n=1- np.exp(-T*phiun)
    def integrand(x):
        if (x<1):
            return(mup(x)*np.exp(-T*(Lap(x)-Mup(x))))
        else:
            return(mup(x-1)*np.exp(-T*(Lap(x-1)-Mup(x-1)+phiun)))
    def f(t):
        a=quad(integrand,t,t+1,limit=limit)[0]
        d=np.exp(T*(Lap(t) -Mup(t)))*a*T
        return (n/(n+d))
    return(f) #on retourne une fonction t -> pe(tT,T) pour t dans (0,1)

def testcalculepe(gamma=0.3,lzero=1.5,alpha=0.5,T=100):
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def la(t):
        if (t< 1-gamma):
            return(lzero)
        else:
            return(lzero*alpha)
    def La(t):
        if (t< 1-gamma):
            return(lzero*t)
        else:
            return(lzero*(1-gamma) + (t-(1-gamma))*lzero*alpha)

    def lam(t,lzero=1.5):
        return(lzero*(1+ np.sin(2*pi*t)))
    def Lam(t,lzero=1.5):
        return(lzero*(t + (1/(2*pi))*(1-np.cos(2*pi*t))))

    npem=nncalculepe(la,La,mu,Mu,T)
    #voirtauxetpe(la,La,mu,Mu,npem,T)
    voirtauxetpe(la,La,mu,Mu,npem,T)

def voirtauxetpe(la,La,mu,Mu,pem,T):
    t=np.linspace(0,1,100)
    plt.rcParams['figure.figsize'] = [16, 10]
    plt.subplot(2,1,1)
    plt.title("taux de naissance et de mort")
    plt.plot(t,[la(s) for s in t],label=r"$\lambda(t)$")
    plt.plot(t,[mu(s) for s in t],label=r"$\mu(t)$")
    plt.legend()

    plt.subplot(2,1,2)
    phiun=La(1)-Mu(1)
    plt.title("probabilite d'extinction, T="+str(T))
    plt.xlabel("instant relatif d'arrivee de l'infecte dans la periode")
    plt.plot(t,[pem(s) for s in t],label=r"$p_e(tT,T)$, T="+str(T))
    plt.plot(t,[(La(s)-Mu(s))/phiun for s in t],label=r"$\varphi(t)/\varphi(1)$")
    plt.legend()

#Regardons maintenant ce que donnent les deux strategies de controle
#totu d'abord le cas ou le taux de croissance est au dessus puis en dessous
# du taux de mort. Ici c'est le cas si lzero*alpha < muzero=1
def prestratb(c,rhom=0.5,gamma=0.3,alpha=0.5,lzero=1.5,T=100):
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def la(t):
        if (t< 1-gamma):
            return(lzero)
        else:
            return(lzero*alpha)
    def La(t):
        if (t< 1-gamma):
            return(lzero*t)
        else:
            return(lzero*(1-gamma) + (t-(1-gamma))*lzero*alpha)

    def frho(t):
        if (t<c):
            return(1-rhom)
        else:
            return(1.0)
    def larho(t):
        return(la(t)*frho(t))
    def Intlarho(t):
            return (rhom*La(min(t,c)))
    def Larho(t):
        return(La(t) -Intlarho(t))
    rzero=Larho(1)/Mu(1)
    assert (La(1)>Mu(1)),r"$R_0$ should be greater than $1$"
    print("Rzero=",La(1)/Mu(1),"Rzero(rhom)=",rzero)
    perho=calculepe(larho,Larho,mu,Mu,T)
    t=np.linspace(0,1,100)
    plt.title(r"probabilite d'extinction : stratégie $\rho_1$")
    plt.xlabel("instant relatif d'arrivee de l'infecte dans la periode")
    plt.plot(t,[perho(s) for s in t],label=r"$p_e(tT,T)$")
    plt.plot(t,[Larho(s) -Mu(s) for s in t],label=r"$\varphi(\rho)$")
    plt.legend()
    pemoyennerho=(quad(perho,0,1)[0])
    print("pemoyenne=",pemoyennerho)
    tstar=(La(1)-Mu(1))/(lzero-1)
    rhomb=rhom*(1-gamma)/tstar
    def frhob(t):
        if (t<tstar):
            return(1-rhomb)
        else:
            return(1.0)
    def larhob(t):
        return(la(t)*frhob(t))
    def Intlarhob(t):
            return (rhomb*La(min(t,tstar)))
    def Larhob(t):
        return(La(t) -Intlarhob(t))
    print("tstar=",tstar,"rhomb=",rhomb,"Rzero(rhob)=",Larhob(1)/Mu(1))
    plt.figure()
    perhob=calculepe(larhob,Larhob,mu,Mu,T)
    t=np.linspace(0,T,100)
    plt.title(r"probabilite d'extinction : stratégie $\rho_2$")
    plt.xlabel("instant relatif d'arrivee de l'infecte dans la periode")
    plt.plot(t,[perhob(s) for s in t],label=r"$p_e(tT,T)$")
    plt.plot(t,[Larhob(s) -Mu(s) for s in t],label=r"$\varphi(\rho)$")
    plt.legend()
    pemoyennerhob=(quad(perhob,0,1)[0])
    print("pemoyennerhob=",pemoyennerhob)
    return(perho,perhob)

def stratb(rhom=0.5,gamma=0.3,alpha=0.5,lzero=1.5,T=100):
    return(prestratb(c=1-gamma,rhom=rhom,gamma=gamma,alpha=alpha,lzero=lzero,T=T))

#stratb(rhom=0.2,T=100)


#regardons les calculs théoriques d'efficacité et vérifiions qu'ils correspondent aux chiffres ci dessus
def efficaciteundeux(rhom=0.2,gamma=0.3,alpha=0.5,lzero=1.5):
    blam=lzero*(1-gamma + alpha*gamma)
    cout=lzero*rhom*(1-gamma)
    numer=(blam -1 -cout)*(blam -1)
    den1=lzero*(blam -1 - (lzero-1)*rhom*(1-gamma))
    erho2=numer/den1
    erho1=(blam -1 -cout)/(lzero*(1-rhom))
    print("Rzero=",blam,"rzero(rho)=",blam-cout)
    return(erho1,erho2)
#efficaciteundeux()


#lmaintenant essayons de trouver la meilleure stratégie à un cout donne
#Regardons maintenant ce que donne une stratégie de prévention brutale.
def opti(rhom=0.2,gamma=0.3,alpha=0.5,lzero=1.5,T=100,N=100):
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def la(t):
        if (t< 1-gamma):
            return(lzero)
        else:
            return(lzero*alpha)
    def La(t):
        if (t< 1-gamma):
            return(lzero*t)
        else:
            return(lzero*(1-gamma) + (t-(1-gamma))*lzero*alpha)
    print("La(1)=",La(1))
    res=np.ones(shape=(2*N,2*N))
    for i in range(N):
        t1=i/N
        for j in range(i+1,i+N):
            t2= j/N
            print("i,j=",i,j)
            if (j <N):
                def frho(t):
                    if ((t1< t) and (t< t2)):
                        return(1-rhom)
                    else:
                        return(1.0)
                def Intlarho(t):
                    return(rhom*(La(min(t,t2)) - La(min(t,t1))))
            else:
                def frho(t):
                    if ((t1< t) or (t< t2-1)):
                        return(1-rhom)
                    else:
                        return(1.0)   
                def Intlarho(t):
                    return(rhom*(La(min(t,1)) - La(min(t,t1)) + La(min(t,t2-1))))
            def larho(t):
                return(la(t)*frho(t))
            def Larho(t):
                return(La(t) -Intlarho(t))
            print("Larho(1)=",Larho(1))
            if (Larho(1)<= Mu(1)):
                res[i,j]=0.0
            else:
                perho=calculepe(larho,Larho,mu,Mu,T)
                t=np.linspace(0,T,100)
                z=[perho(s) for s in t]
                res[i,j]=max(z)
    return(res)




#Mardi 4 décembre 2018
#Regardons maintenant ce que donnent les deux strategies de controle
#totu d'abord le cas ou le taux de croissance est au toujours au dessus 
# du taux de mort. Ici c'est le cas si lzero*alpha > muzero=1
#il n'y a plus de t*, alors il faut le fournir
def stratc(tstar,rhom=0.5,gamma=0.3,alpha=0.5,lzero=4,T=100):
    c=1-gamma
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def la(t):
        if (t< 1-gamma):
            return(lzero)
        else:
            return(lzero*alpha)
    def La(t):
        if (t< 1-gamma):
            return(lzero*t)
        else:
            return(lzero*(1-gamma) + (t-(1-gamma))*lzero*alpha)

    def frho(t):
        if (t<c):
            return(1-rhom)
        else:
            return(1.0)
    def larho(t):
        return(la(t)*frho(t))
    def Intlarho(t):
            return (rhom*La(min(t,c)))
    def Larho(t):
        return(La(t) -Intlarho(t))
    rzero=Larho(1)/Mu(1)
    assert (La(1)>Mu(1)),r"$R_0$ should be greater than $1$"
    print("Rzero=",La(1)/Mu(1),"Rzero(rhom)=",rzero)
    perho=newcalculepe(larho,Larho,mu,Mu,T,limit=100)
    pemoyennerho=(quad(perho,0,1)[0])
    print("pemoyenne=",pemoyennerho)
    abc=np.arange(0,1,1/100) #les abcissess des points
    plt.title(r"probabilite d'extinction : stratégie $\rho_1$,$\lambda_0=$"+str(lzero)+r", $\alpha=$" +str(alpha) + r", $\rho_M=$"+str(rhom)+r" efficacite="+str(pemoyennerho))
    plt.xlabel("instant relatif d'arrivee de l'infecte dans la periode")
    #plt.plot(abc,[Intlarho(s) for s in abc],label=r"$\int_0^t \rho(s) \lambda(s) ds$")
    plt.plot(abc,[perho(s) for s in abc],label=r"$p_e(tT,T)$")
    plt.plot(abc,[Larho(s) -Mu(s) for s in abc],label=r"$\varphi_{\rho_1}$")
    plt.plot(abc,[larho(s)/lzero for s in abc],label=r"$\lambda_{\rho_1}(t)/\lambda_0$")
    plt.legend()
    #tstar=(La(1)-Mu(1))/(lzero-1) #on ne le calcule pas c'est un parametre
    rhomb=rhom*(1-gamma)/tstar
    def frhob(t):
        if (t<tstar):
            return(1-rhomb)
        else:
            return(1.0)
    def larhob(t):
        return(la(t)*frhob(t))
    def Intlarhob(t):
            return (rhomb*La(min(t,tstar)))
    def Larhob(t):
        return(La(t) -Intlarhob(t))
    print("tstar=",tstar,"rhomb=",rhomb,"Rzero(rhob)=",Larhob(1)/Mu(1))
    plt.figure()
    perhob=newcalculepe(larhob,Larhob,mu,Mu,T,limit=100)
    pemoyennerho2=(quad(perhob,0,1)[0])
    print("pemoyennerho2=",pemoyennerho2)
    plt.title(r"probabilite d'extinction : stratégie $\rho_2$, $t^*=$"+str(tstar)+r" efficacite="+str(pemoyennerho2))
    plt.xlabel("instant relatif d'arrivee de l'infecte dans la periode")
    plt.plot(abc,[perhob(s) for s in abc],label=r"$p_e(tT,T)$")
    plt.plot(abc,[Larhob(s) -Mu(s) for s in abc],label=r"$\varphi_{\rho_2}$")
    plt.plot(abc,[larhob(s)/lzero for s in abc],label=r"$\lambda_{\rho_2}(t)/\lambda_0$")

    plt.legend()
    return(perho,perhob)


#mercredi 12 decembre 2018. Regardons la strategie de controle qui parait idiote, qui est de baisser lambda sur (1-gamma,1), dans le cas ou lambda est au desssus et au dessous de mu
def stratdebile(rhom=0.5,gamma=0.3,alpha=0.5,lzero=1.5,T=100):
    c=1-gamma
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def la(t):
        if (t< 1-gamma):
            return(lzero)
        else:
            return(lzero*alpha)
    def La(t):
        if (t< 1-gamma):
            return(lzero*t)
        else:
            return(lzero*(1-gamma) + (t-(1-gamma))*lzero*alpha)

    def frho(t):
        if (t<c):
            return(1-rhom)
        else:
            return(1.0)
    def larho(t):
        return(la(t)*frho(t))
    def Intlarho(t):
            return (rhom*La(min(t,c)))
    def Larho(t):
        return(La(t) -Intlarho(t))
    rzero=Larho(1)/Mu(1)
    assert (La(1)>Mu(1)),r"$R_0$ should be greater than $1$"
    print("Rzero=",La(1)/Mu(1),"Rzero(rhom)=",rzero)
    voirprobext(larho,Larho,mu,Mu,T,nom=r"stratégie $\rho_1$")
    
    tstar=(La(1)-Mu(1))/(lzero-1)
    # rhomb=rhom*(1-gamma)/tstar
    # def frhob(t):
    #     if (t<tstar):
    #         return(1-rhomb)
    #     else:
    #         return(1.0)
    # def larhob(t):
    #     return(la(t)*frhob(t))
    # def Intlarhob(t):
    #         return (rhomb*La(min(t,tstar)))
    # def Larhob(t):
    #     return(La(t) -Intlarhob(t))
    # print("tstar=",tstar,"rhomb=",rhomb,"Rzero(rhob)=",Larhob(1)/Mu(1))
    # plt.figure()
    # perhob=newcalculepe(larhob,Larhob,mu,Mu,T,limit=100)
    # plt.title(r"probabilite d'extinction : stratégie $\rho_2$")
    # plt.xlabel("instant relatif d'arrivee de l'infecte dans la periode")
    # plt.plot(t,[perhob(s) for s in t],label=r"$p_e(tT,T)$")
    # plt.plot(t,[Larhob(s) -Mu(s) for s in t],label=r"$\varphi(\rho)$")
    # plt.legend()
    # pemoyennerhob=(quad(perhob,0,1)[0])
    # print("pemoyennerhob=",pemoyennerhob)
    # #maintenant on baisse sur (c=1-gamma,1)
    rhomdeb=rhom*(1-gamma)/((1-c)*alpha)
    def frhodeb(t):
        if (t>c):
            return(1-rhomdeb)
        else:
            return(1.0)
    def larhodeb(t):
        return(la(t)*frhodeb(t))
    def Intlarhodeb(t):
        if (t< c):
            return(0.0)
        else:
            return (rhomdeb*(La(t)-La(c)))
    def Larhodeb(t):
        return(La(t) -Intlarhodeb(t))
    print("tstar=",tstar,"rhomdeb=",rhomdeb,"Rzero(rhodeb)=",Larhodeb(1)/Mu(1))
    voirprobext(larhodeb,Larhodeb,mu,Mu,T,nom=r"stratégie $\rho_s$")


#stratdebile(rhom=0.2,T=100)


def voirprobext(la,La,mu,Mu,T,nom=""):
    t=np.arange(0,1,1/100)
    plt.figure()
    pe=newcalculepe(la,La,mu,Mu,T,limit=100)
    pemoyenne=(quad(pe,0,1)[0])
    print("voirprobext: pemoyenne=",pemoyenne)
    # plt.title(r"probabilite d'extinction : " + nom +" moyenne=",pemoyenne)
    plt.xlabel("instant relatif d'arrivee de l'infecte dans la periode")
    plt.plot(t,[pe(s) for s in t],label=r"$p_e(tT,T)$")
    plt.plot(t,[La(s) -Mu(s) for s in t],label=r"$\varphi$")
    plt.legend()

#jeudi 13 decembre. Calcul directement de la mpyenne de la probabiite d'extinction dans le cas ou lambda est un creneau simple : ne prend que les valeurs lambda_0 et 0

def theoeff(cout=0.2,gamma=0.3,lzero=2,tun=0,tdeux=0.5):
    r""" calcul de l'efficacite theorique"""
    #on verifie que tout va bien
    phiun=lzero*(1-gamma) -1
    deltat=tdeux-tun
    assert (phiun>0) and (cout < phiun)
    assert (tun <= tdeux) and (tdeux <= 1-gamma)
    assert ((cout/lzero) + deltat <= 1-gamma)
    phirhoun=phiun-cout
    rhosm=(cout/(lzero*(1-gamma-deltat)))
    phirhotun=tun*(lzero*(1-rhosm)-1)
    phirhotdeux=phirhotun+deltat*(lzero-1)
    print("phiroun,rhosm,phirhotun,phirhotdeux=",phirhoun,rhosm,phirhotun,phirhotdeux)
    if (phirhotun<=0):
        return(phirhoun/lzero)
    elif (phirhoun < phirhotun):
        return(phurhoun/(lzero*(1-rhosm)))
    elif (phirhoun < phirhotdeux):
        return((phirhoun+phirhotun*(rhosm/(1-rhosm)))/lzero)
    else:
        return((phirhoun+(phirhotdeux-phirhotun)/(lzero*(1-rhosm)-1))/(lzero*(1-rhosm)))
    

def illutheoeff(cout=0.2,gamma=0.3,lzero=2,tun=0,tdeux=0.5,T=100):
    etheo=theoeff(cout=cout,gamma=gamma,lzero=lzero,tun=tun,tdeux=tdeux)
    #c=1-gamma
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def la(t):
        if (t< 1-gamma):
            return(lzero)
        else:
            return(0.0)
    def La(t):
        if (t< 1-gamma):
            return(lzero*t)
        else:
            return(lzero*(1-gamma))
    
    phiun=lzero*(1-gamma) -1
    deltat=tdeux-tun
    assert (phiun>0) and (cout < phiun)
    assert (tun <= tdeux) and (tdeux <= 1-gamma)
    assert ((cout/lzero) + deltat <= 1-gamma)
    phirhoun=phiun-cout
    rhosm=(cout/lzero)/(1-gamma-deltat)

    def frho(t):
        if ((t< tun) or (tdeux< t< 1-gamma)):
            return(1-rhosm)
        else:
            return(1.0)
    def larho(t):
        return(la(t)*frho(t))
    def Intlarho(t):
        if (t < tun):
            return(t*lzero*(rhosm))
        elif (t< tdeux):
            return(tun*lzero*(rhosm))
        elif (t< 1-gamma):
            return((tun+(t-tdeux))*lzero*(rhosm))
        else:
            return((tun+(1-gamma-tdeux))*lzero*(rhosm))
                   
    def Larho(t):
        return(La(t) -Intlarho(t))
    print(Larho(1),Intlarho(1))
    print("phirhotun,phirhotdeux,phirhoun",Larho(tun)-Mu(tun),Larho(tdeux)-Mu(tdeux),Larho(1)-Mu(1))
    print("pemoyennetheorique=",etheo)
    voirprobext(larho,Larho,mu,Mu,T=T)
    plt.figure()
    t=np.arange(0,1,1/100)
    plt.plot(t,[larho(s) for s in t],label=r"$\lambda(t)$")
    plt.legend()


#vendredi 14 decembre 2018. Comparaison de la strategie naive et de la strategie optimale
def compstrat(lzero=2.0,gamma=0.3,beamer=False):
    phiun=lzero*(1-gamma)-1
    assert (phiun>0)
    t=np.arange(0,phiun,1/100)#les valeurs prises par le cout
    #print(phiun,t)
    def couopt(c):
        return((phiun-c)/lzero)
    def counaif(c):
        return(couopt(c)/(1- c/(lzero*(1-gamma))))
    if (beamer):
        plt.rcParams['figure.figsize'] = [14, 10]
    else:
        plt.rcParams['figure.figsize'] = [10,14]
        
    #plt.title("mean emergence probability for the step example :"+ r"$\lambda_0=$"+str(lzero)+r", $\gamma=$"+str(gamma))
    plt.xlabel("cost")
    plt.ylabel("mean emergence probability",fontsize=16)
    plt.plot(t,[couopt(s) for s in t],dashes=[2,2],color='black', label=r"optimal")
    plt.plot(t,[counaif(s) for s in t], color='black', label=r"naive")
    plt.legend()
    
    plt.savefig("compstratstep.pdf",dpi=300)

    
def ncompstrat(lzero=2.0,rzero=3.0,beamer=False):
    assert (rzero>1)
    cmax=(rzero -1)/lzero
    #t=np.arange(0,cmax,1/100)#les valeurs prises par le cout
    t=np.linspace(0,cmax,100)#les valeurs prises par le cout
    #print(phiun,t)
    def couopt(c):
        return((rzero -1 - c*lzero)/lzero)
    def counaif(c):
        return(couopt(c)*rzero/(rzero - c*lzero))
    if (beamer):
        plt.rcParams['figure.figsize'] = [16, 10]
    else:
        plt.rcParams['figure.figsize'] = [10,16]
        
    #plt.title("mean emergence probability for the step example :"+ r"$\lambda_0=$"+str(lzero)+r", $\gamma=$"+str(gamma))
    plt.xlabel(r"\bf{Cost of control,} ${C}$",fontsize=30)
    plt.ylabel(r"\bf{Mean emergence probability} ${<p_{e,\rho,\infty}>}$",fontsize=30)
    plt.yticks((0.0, 0.2,0.4,0.6,0.8), (r'\bf{0}', r'\bf{0.2}', r'\bf{0.4}',r'\bf{0.6}',r'\bf{0.8}'), color='k', size=20)
    plt.xticks((0.0, 0.2,0.4,0.6,0.8,1.0), (r'\bf{0}', r'\bf{0.2}', r'\bf{0.4}',r'\bf{0.6}',r'\bf{0.8}',r'\bf{1.0}'), color='k', size=20)
    plt.plot(t,[couopt(s) for s in t],dashes=[2,2],color='black', label=r"optimal")
    plt.plot(t,[counaif(s) for s in t], color='black', label=r"naive")
    aprest=np.arange(cmax,1.5*cmax,1/100)
    plt.plot(aprest,[0.0 for s in aprest],color='black')
    plt.text(cmax+0.01,0.05,r"${\frac{R_0 -1}{\lambda_0}}$",fontsize=30)

    #plt.legend()
    
    plt.savefig("ncompstratstep.pdf",dpi=300)


               
    
#Lundi 17 decembre essayons de calculer numeriquement

def casin(lzero=2.0,T=100,C=0.2):
    def la(t):
        return(lzero*(1+np.sin(2*pi*t)))
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def La(t):
        return(lzero*(t + (1/(2*pi))*(1-np.cos(2*pi*t))))
    
    def sinqextmoy(tun,tdeux,C=C,voir=False):
        assert (tun >=0)
        if (tun >= 0.5):
            return(0.0)
        if (tdeux <= tun):
            return(0.0)
        deltat=tdeux -tun
        if (deltat < C):
            return(0.0)
        rhom=C/deltat
        def larho(t):
            f=(1-rhom) if ((tun<t) and (t< tdeux)) else 1.0
            return(f*la(t))
        Latun=La(tun)
        Latdeux=La(tdeux)
        Larhotdeux=Latun + (1-rhom)*(Latdeux-Latun)
        #print(rhom,Latun,Latdeux,Larhotdeux)
        def Larho(t):
            if (t<= tun):
                return(La(t))
            elif (t <= tdeux):
                return(Latun + (1-rhom)*(La(t) -Latun))
            else:
                return(Larhotdeux +La(t) -La(tdeux))
        pe=nncalculepe(larho,Larho,mu,Mu,T,limit=100)
        if voir:
            t=np.arange(0,1,1/100)
            plt.plot(t,[Larho(s) for s in t],label=r"$\Lambda_\rho$")
            plt.plot(t,[pe(s) for s in t],label=r"$p_e$")
            plt.legend()
        return (1-(quad(pe,0,1)[0]))
    #q=sinqextmoy(0.1,0.4,0.2)
    #print("q=",q)
    return(sinqextmoy)

from mpl_toolkits.mplot3d import Axes3D

def vcasin(nbpt):
    f=casin(T=50)
    x=np.arange(0,0.5,1.0/nbpt)
    zs = np.array([[f(i,j) for j in x] for i in x])
    return(zs)

def simplevoir(Z,nbpt):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x=np.arange(0,0.5,1.0/nbpt)
    y=np.arange(0,0.5,1.0/nbpt)
    X,Y=np.meshgrid(x,y)
    ax.plot_surface(X, Y, Z)
    #ax.contour3D(X,Y,Z,50,cmap='binary')

    ax.set_xlabel(r"$t_1$")
    ax.set_ylabel(r"$t_2$")
    ax.set_zlabel(r"mean extinction probability")

    plt.show()


def nvoir(z):
    N=(z.shape)[0]
    data=pd.DataFrame(z)
    df=data.unstack().reset_index()
    df.columns=["X","Y","Z"]
    df['X']=df['X']*0.5/N
    df['Y']=df['Y']*0.5/N
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
    ax.set_xlabel(r"$t_1$")
    ax.set_ylabel(r"$t_2$")
    ax.set_zlabel(r"mean extinction probability")
    #ax.view_init(30, 45)
    ax.set_zlim(0.8,1.0)
def nnvoir(z,C=0.2):
    N=(z.shape)[0]
    tun=np.arange(0.0,1.0,1/N)
    X,Y=np.meshgrid(tun,(1.0/N)+tun)
    fig, ax = plt.subplots()
    #print(tun.shape,X.shape,z.shape)
    #levels = np.arange(0.04, 0.2, 0.02)
    #CS = ax.contour(X,Y,z,levels,cmap='flag')
    CS = ax.contour(X,Y,z,cmap='flag')
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_ylabel(r"$t_1$")
    ax.set_xlabel(r"$\rho_M$")
    ax.set_title('Mean emergence probability: Contour Plot. Cost='+str(C))
    plt.show()

        
#mardi 18 decembre : il vaut mieux garder comme parametres d'une strategie t_1 et rho_M
def rcasin(lzero=2.0,T=100,C=0.2):
    def la(t):
        return(lzero*(1+np.sin(2*pi*t)))
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def La(t):
        return(lzero*(t + (1/(2*pi))*(1-np.cos(2*pi*t))))
    
    def sinqextmoy(tun,rhom,C=C,voir=False):
        #assert (tun >=0)
        #assert (rhom>0) and (rhom<=1)
        deltat=min(C/rhom,1-tun)
        tdeux=tun+deltat
        def larho(t):
            f=(1-rhom) if ((tun<t) and (t< tdeux)) else 1.0
            return(f*la(t))
        Latun=La(tun)
        Latdeux=La(tdeux)
        Larhotdeux=Latun + (1-rhom)*(Latdeux-Latun)
        #print(rhom,Latun,Latdeux,Larhotdeux)
        def Larho(t):
            if (t<= tun):
                return(La(t))
            elif (t <= tdeux):
                return(Latun + (1-rhom)*(La(t) -Latun))
            else:
                return(Larhotdeux +La(t) -La(tdeux))
        pe=nncalculepe(larho,Larho,mu,Mu,T,limit=100)
        if voir:
            t=np.arange(0,1,1/100)
            plt.plot(t,[Larho(s) for s in t],label=r"$\Lambda_\rho$")
            plt.plot(t,[pe(s) for s in t],label=r"$p_e$")
            plt.legend()
        return (1-(quad(pe,0,1)[0]))
    #q=sinqextmoy(0.1,0.4,0.2)
    #print("q=",q)
    return(sinqextmoy)


def rvcasin(nbpt,C):
    f=rcasin(T=50,C=C)
    x=np.arange(0,1.0,1.0/nbpt)
    rho=(1.0/nbpt)+x
    zs = np.array([[f(i,j) for j in rho] for i in x])
    return(zs)

def testrvcasin(N,C):
    z=rvcasin(N,C=C)
    np.save("testrvcasinN"+str(N)+"C"+str(C),z)
    return(z)


#mardi 18 decembre : idee du calcul direct de p_{e,infty}
def peinfmoy(la,La,mu,Mu,N=100,voir=False):
    phiun=La(1)-Mu(1)
    def phi(t):
        if (t<=1):
            return(La(t)-Mu(t))
        if (t<=2):
            return(phiun+La(t)-Mu(t))

    
    lesphi=np.array([phi(i/N) for i  in np.arange(2*N+1)])
    def peinf(i):
        t=i/N
        #print(i,t,la(t),mu(t),La(t),Mu(t))
        if (la(t) <= mu(t)):
            return(0.0)
        z=lesphi[i+1:i+N]- lesphi[i]
        #print("z=",z)
        if (z.min()<=0):
            return(0.0)
        else:
            return(1-(mu(t)/la(t)))
    ab=np.arange(N+1)
    val=np.array([peinf(i) for i in ab])
    if voir:
        plt.plot(ab/N,val,label=r"$p_{e,\infty}$")
        plt.legend()
    #on fait une integration par la methode des trapezes
    return(np.trapz(val,ab/N))

pi=np.pi
def pemsin(lzero=2.0,N=100):
    r"""Calcule la probabilite d'emergence moyenne, dans le cas sinusoidal, a l'infini, sans controle"""
    def la(t):
        return(lzero*(1+np.sin(2*pi*t)))
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def La(t):
        return(lzero*(t + (1/(2*pi))*(1-np.cos(2*pi*t))))
    reuurn(speinfmoy(la,La,mu,Mu,N=N))

def vpemsin(lzero=2.0,T=50):
    r"""" Represente la probabilite d'emergence a horizon fini pour la fonction de taux sinusoidale"""
    def la(t):
        return(lzero*(1+np.sin(2*pi*t)))
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def La(t):
        return(lzero*(t + (1/(2*pi))*(1-np.cos(2*pi*t))))

    pe=nncalculepe(la,La,mu,Mu,T,limit=100)
    t=np.arange(0,1,1/100)
    plt.plot(t,[pe(s) for s in t],label=r"$p_e$")
    plt.plot(t,[La(s)-Mu(s) for s in t],label=r"$\varphi$")
    plt.xlabel("Introduction time of the infected")
    plt.legend()
    m=quad(pe,0,1)[0]
    titre=r"Sinusoidal birth rate. Emergence probability. $\lambda_0$="+str(lzero)+r" $T=$"+str(T) +r"$<p_e>$="+'{:.4f}'.format(m)
    plt.title(titre)



           
    
def simusin(lzero=2.0,C=0.2):
    def la(t):
        return(lzero*(1+np.sin(2*pi*t)))
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def La(t):
        return(lzero*(t + (1/(2*pi))*(1-np.cos(2*pi*t))))
    
    def sinqextmoy(tun,rhom,C=C,voir=False):
        deltat=min(C/rhom,1-tun)
        tdeux=tun+deltat
        def larho(t):
            f=(1-rhom) if ((tun<t) and (t< tdeux)) else 1.0
            return(f*la(t))
        Latun=La(tun)
        Latdeux=La(tdeux)
        Larhotdeux=Latun + (1-rhom)*(Latdeux-Latun)
        def Larho(t):
            if (t<= tun):
                return(La(t))
            elif (t <= tdeux):
                return(Latun + (1-rhom)*(La(t) -Latun))
            else:
                return(Larhotdeux +La(t) -La(tdeux))
        return(peinfmoy(larho,Larho,mu,Mu))

    return(sinqextmoy)

def grillesimusin(nbpt,C=0.2,tunmax=0.5):
    f=simusin(C=C)
    tun=np.arange(0,tunmax,tunmax/nbpt)
    rho=(1.0/nbpt)+np.arange(0,1.0,1.0/nbpt)
    zs = np.array([[f(i,j) for j in rho] for i in tun])
    return(zs)

def testsimusin(nbpt=100,C=0.3):
    z=grillesimusin(nbpt=nbpt,C=C)
    np.save("testsimusinN"+str(nbpt)+"C"+str(C),z)
    return(z)

def sinvoir(z,C=0.2,tunmax=0.5):
    N=(z.shape)[0]
    tun=np.arange(0.0,tunmax,tunmax/N)
    X,Y=np.meshgrid((1.0/N)+np.arange(0.0,1.0,1/N),tun)
    fig, ax = plt.subplots()
    #levels = np.arange(0.04, 0.2, 0.02)
    #CS = ax.contour(X,Y,z,levels,cmap='flag')
    CS = ax.contour(X,Y,z,cmap='flag')
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_ylabel(r"$t_1$")
    ax.set_xlabel(r"$\rho_M$")
    m=z.min()
    y=np.where(z==m)
    xopt=(y[0][0]/N)*tunmax
    yopt=y[1][0]/N
    titre=r"Mean emergence probability: Contour Plot. Cost="+str(C)+"\n"+ r""" minimal $<p_e>$=""" +'{:.4f}'.format(m)+r""" for $t_1$=""" + str(xopt)+r""", $\rho_M=$"""+str(yopt)
    ax.set_title(titre)
    plt.show()


def grillerzerosin(nbpt,C=0.2,tunmax=0.5):
    f=rzerosin(C=C)
    tun=np.arange(0,tunmax,tunmax/nbpt)
    rho=(1.0/nbpt)+np.arange(0,1.0,1.0/nbpt)
    zs = np.array([[f(i,j) for j in rho] for i in tun])
    return(zs)

def testrzerosin(nbpt=100,C=0.2):
    z=grillerzerosin(nbpt=nbpt,C=C)
    np.save("testrzerosinN"+str(nbpt)+"C"+str(C),z)
    return(z)

def rzerosinvoir(z,C=0.2,tunmax=0.5):
    N=(z.shape)[0]
    tun=np.arange(0.0,tunmax,tunmax/N)
    X,Y=np.meshgrid((1.0/N)+np.arange(0.0,1.0,1/N),tun)
    fig, ax = plt.subplots()
    levels = np.arange(1.6, 1.8, 0.001)
    CS = ax.contour(X,Y,z,levels=levels,cmap='flag')
    #CS = ax.contour(X,Y,z,cmap='flag')
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_ylabel(r"$t_1$")
    ax.set_xlabel(r"$\rho_M$")
    m=z.min()
    y=np.where(z==m)
    xopt=(y[0][0]/N)*tunmax
    yopt=y[1][0]/N
    titre=r"Basic Reproduction Number: Contour Plot. Cost="+str(C)+"\n"+ r""" minimal $R_0$=""" +'{:.4f}'.format(m)+r""" for $t_1$=""" + str(xopt)+r""", $\rho_M=$"""+str(yopt)
    ax.set_title(titre)
    plt.show()

def rzerosin(lzero=2.0,C=0.2):
    def la(t):
        return(lzero*(1+np.sin(2*pi*t)))
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def La(t):
        return(lzero*(t + (1/(2*pi))*(1-np.cos(2*pi*t))))
    
    def rzerorho(tun,rhom,C=C):
        deltat=min(C/rhom,1-tun)
        tdeux=tun+deltat
        return((La(1) -rhom*(La(tdeux)-La(tun)))/Mu(1))

    return(rzerorho)

########## pour le cas step, on visualise a la fois plusieurs periodes
def stepvoirphietpe(gamma=0.3,lzero=4.0,lT=[20,50,100]):
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def la(t):
        if (t< 1-gamma):
            return(lzero)
        else:
            return(0.0)
    def La(t):
        if (t< 1-gamma):
            return(lzero*t)
        else:
            return(lzero*(1-gamma))

    t=np.linspace(0,1,100)
    #plt.rcParams['figure.figsize'] = [16, 10]
    plt.subplot(2,1,1)
    plt.title("Birth and Death rates")
    plt.plot(t,[la(s) for s in t],label=r"$\lambda(t)$")
    plt.plot(t,[mu(s) for s in t],label=r"$\mu(t)$")
    plt.legend()

    plt.subplot(2,1,2)
    plt.xlabel("Introduction time of the infected")

    phiun=La(1)-Mu(1)
    plt.title("Extinction probability")
    for T in lT:
        pem=nncalculepe(la,La,mu,Mu,T)
        plt.plot(t,[pem(s) for s in t],label=r"$p_e(tT,T)$, T="+str(T))
    plt.plot(t,[(La(s)-Mu(s))/phiun for s in t],label=r"$\varphi(t)/\varphi(1)$")
    plt.legend()


################ jeudi 24 janvier 2019
#from scipy.integrate import odeint
#tout d'abord ecrire la fonction a optimiser
def lafoncao(l12,mu1,l21,mu2,T,nbpts=200):
    def msisi(x,t):
        return([(l12(t)+mu1(t))*x[0]-(mu1(t)+l12(t)*x[0]*x[1]),
                (l21(t)+mu2(t))*x[1]-(mu2(t)+l21(t)*x[0]*x[1])])
    def f(y):
        timeint=np.arange(0,T+1/nbpts,T/nbpts)
        z=np.array(odeint(msisi,y,timeint))
        fin=z[-1]
        print("y1,y2=",y," z=",z)
        plt.plot(timeint,z[:,0],label="y1")
        plt.plot(timeint,z[:,1],label="y2")
        plt.legend()
        r=fin-y
        print("r=",r,"norme(r)carree=",(r*r).sum())
        return((r*r).sum())
    return(f)

def l21(x):
    return(1/8)
def l12(x):
    return(0.22)
def mu1(x):
    return(1/5)
def mu2(x):
    return(1/8)
f=lafoncao(l12,mu1,l21,mu2,1)

#la solution stationnaire, quand les fonctions sont constantes
def sstat(l12,mu1, l21,mu2):
    x=1
    return( [1- (l12(x)*l21(x)-mu1(x)*mu2(x))/(l21(x)*(l12(x)+mu1(x))),
    1- (l12(x)*l21(x)-mu1(x)*mu2(x))/(l12(x)*(l21(x)+mu2(x)))])
     

def champvec(l12,mu1, l21,mu2):
    def f(x):
        t=1
        return(np.array([(l12(t)+mu1(t))*x[0]-(mu1(t)+l12(t)*x[0]*x[1]),
                (l21(t)+mu2(t))*x[1]-(mu2(t)+l21(t)*x[0]*x[1])]))
    return(f)
y=sstat(l12,mu1, l21,mu2)
F=champvec(l12,mu1, l21,mu2)
F([1,1]),F(y) #les deux doivent donner 0
#revenons au test
#f([0.2,0.3])


#on observe meme pour des coeff constants un systeme de lotka volterra avec un equilibre instable

#vendredi 25 janvier. Utilisons l'approche de Bacaer
def oldfbaca(l12,mu1,l21,mu2,T,tau,nbpts=200):
    def msisi(x,t):
        return([(l12(t)+mu1(t))*x[0]-(mu1(t)+l12(t)*x[0]*x[1]),
                (l21(t)+mu2(t))*x[1]-(mu2(t)+l21(t)*x[0]*x[1])])
    def f(y):
        timeint=np.arange(0,T+1/nbpts,T/nbpts)
        z=np.array(odeint(msisi,y,timeint))
        fin=z[-1]
        print("y1,y2=",y," z=",z)
        plt.plot(timeint,z[:,0],label="y1")
        plt.plot(timeint,z[:,1],label="y2")
        plt.legend()
        r=fin-y
        print("r=",r,"norme(r)carree=",(r*r).sum())
        return((r*r).sum())
    return(f)


#lundi 28 janvier, on ecrit carrement les equations verifiees par z
#on resout l'equation (8) D Bacaer
def fbaca(l12,mu1,l21,mu2,tau,T,nbpts=200,zzero=[1.0,1.0],tv=False):
    def msisi(x,t):
        s=(tau-t)/T
        return([-mu1(s)*x[0] +l12(s)*x[1]*(1-x[0]),
        -mu2(s)*x[1] +l21(s)*x[0]*(1-x[1])])

    timeint=np.arange(0,tau+1/nbpts,tau/nbpts)
    z=np.array(odeint(msisi,zzero,timeint))
    #print("T=",T,"Z.shape",z.shape,"z=",z)
    i=int(T*nbpts/tau)
    res=z[-i:]
    if tv:
        t=np.arange(1/nbpts,T+1/nbpts,T/nbpts)
        plt.plot(timeint,z,label="z")
    return(res[::-1])
def fvraipe(l12,mu1,l21,mu2,T,nbpts=200,zzero=[1.0,1.0]):
    def msisi(x,t):
        s=t/T
        return([mu1(s)*x[0] -l12(s)*x[1]*(1-x[0]),
        mu2(s)*x[1] -l21(s)*x[0]*(1-x[1])])

    timeint=np.arange(0,T+1/nbpts,T/nbpts)
    z=np.array(odeint(msisi,zzero,timeint))
    return(z)
    
 #la solution stationnaire, quand les fonctions sont constantes, pour les z
def zstat(l12,mu1, l21,mu2):
    x=1
    return( [(l12(x)*l21(x)-mu1(x)*mu2(x))/(l21(x)*(l12(x)+mu1(x))),
    (l12(x)*l21(x)-mu1(x)*mu2(x))/(l12(x)*(l21(x)+mu2(x)))])
     
def zchampvec(l12,mu1, l21,mu2):
    def f(x):
        t=1
        tau=100
        return(np.array([-mu1(tau-t)*x[0] +l12(tau-t)*x[1]*(1-x[0]),
        -mu2(tau-t)*x[1] +l21(tau-t)*x[0]*(1-x[1])]))
    return(f)

def testonsfbacaetfvraipe():
    #on a bien convergence vers la solution constante, quand les coeff le sont
    z=fbaca(l12,mu1,l21,mu2,tau=1000,T=100,nbpts=10000)
    fin=z[-1]
    g=zchampvec(l12,mu1, l21,mu2)
    vraisol=zstat(l12,mu1, l21,mu2)
    fin,g(fin),vraisol,g(vraisol) #presque 0.0, on est proche de la solution 
    #et si on lance l'ode depuis cette solution approchee cela pose probleme
    zz=fvraipe(l12,mu1,l21,mu2,T=100,nbpts=200,zzero=fin)
    zzz=fvraipe(l12,mu1,l21,mu2,T=100,nbpts=200,zzero=vraisol)
    plt.plot(zz)
    plt.plot(zzz)

    #alors que si on lance l'ode deouis un autre point cela explose
    z4=fvraipe(l12,mu1,l21,mu2,T=100,nbpts=200,zzero=[0.1,0.2])

#c'est normal si on calcule le Jacobien au point d'équilibre on trouve la trac et le determinant >0 donc le systeme est hyperbolique : le point fixe est instable.
#introuduisons maintenant un coeff periodique

def tperfbaca(ldu=1,ab=2.0,ep=1,muu=1,mud=1,tau=300,T=50,nb=10):
    def l21(x):
        return(ldu)
    def l12(x):
        return(ab*(1+ep*np.sin(2*np.pi*x)))
    def mu1(x):
        return(muu)
    def mu2(x):
        return(mud)
    z=fbaca(l12,mu1,l21,mu2,tau=tau,T=T,nbpts=tau*nb,tv=True)

#tperfbaca(ab=2.0)

#comparons avec l'approximation fast/slow
def fslow(l12,mu1,l21,mu2,T,limit=50):
    def mu(x):
        return(mu1(x))
    def Mu(x):
        return(x *mu1(x)) #seule la fonctionl12 n'est pas constante
    def la(x):
        return(l21(x)*l12(x)/mu2(x))
    def La(x):
        return(quad(la,0,x,limit=limit)[0])
    pem=nncalculepe(la,La,mu,Mu,T)
    t=np.linspace(0,1,100)
    return(np.array([pem(s) for s in t]))

#introuduisons maintenant un coeff periodique
def pexper(ldu=1/8,ab=0.22,ep=0.3,muu=1/25,mud=1/8,tau=300,T=50,nb=10):
    def l21(x):
        return(ldu)
    def l12(x):
        return(ab*(1+ep*np.sin(2*np.pi*x)))
    def mu1(x):
        return(muu)
    def mu2(x):
        return(mud)
    def cible1(x):
        z=(l12(x)*l21(x)-mu1(x)*mu2(x))/(l21(x)*(l12(x)+mu1(x)))
        return(z if (z>=0) else 0.0)
    z=fbaca(l12,mu1,l21,mu2,tau=tau,T=T,nbpts=tau*nb)
    plt.plot(np.linspace(0,1,z.shape[0]),z[:,0],label="one infected human")
    plt.plot(np.linspace(0,1,z.shape[0]),z[:,1],label="one infected vector")
    #zz=fslow(l12,mu1,l21,mu2,T=T,limit=50)
    #plt.plot(np.linspace(0,1,zz.shape[0]),zz,label="fast/slow approximation (human)")
    tt=np.linspace(0,1,100)
    tz=np.array([cible1(x) for x in tt])
    plt.plot(tt,tz,label="asymptotic profile for one infected human")
    plt.title("Emergence probability")
    plt.xlabel("time of introduction")
    plt.legend()




#mettons les memes parametres en moyenne excepte pour le parametre oscillant
def testonspexper():
    pexper(muu=1,ldu=1,ab=2.0,ep=1,mud=1,tau=500,T=100) #ca colle rzero=2
    
    plt.figure()
    pexper(muu=1,ldu=1,ab=1.2,ep=1,mud=1,tau=500,T=100)  #ca ne arche pas car le rzero du system BD de dimension 2 est inferieur a 1, alors que le Rzero du systeme 1d BD approximant est superieueur a 1, ici rzero=1.2 (en prenant les valeurs moyennes : je ne sais pas l'approcher comme le fait Bacaer , je prends la formule du rzero pour coeff constants et je remplace les coeff par leur valeur moyenne).


    
#on s'appercoit que l'on peut avoir des probabilites d'emergence quasi nulles avec ab ldu - mud muu >0 : ce n'est donc pas le bon critère pour avoir Rzero>1
#on a un decalage certain avec l'approximation fast/slow

#il reste a verifier que la fonction proba d'emergence periodique donnee par la'lgorithme de Bacaer est effectivement une solution periodique du systeme differentiel calcule pour les proba d'emergence
def checkbaca(ldu=1,ab=2.0,ep=1.0,muu=1,mud=1,tau=300,T=50,nb=10):
    def l21(x):
        return(ldu)
    def l12(x):
        return(ab*(1+ep*np.sin(2*np.pi*x)))
    def mu1(x):
        return(muu)
    def mu2(x):
        return(mud)
    z=fbaca(l12,mu1,l21,mu2,tau=tau,T=T,nbpts=tau*nb)
    plt.plot(np.linspace(0,1,z.shape[0]),z,label="2d approximation")
    zz=fvraipe(l12,mu1,l21,mu2,T=T,nbpts=T*nb,zzero=z[0])
    plt.plot(np.linspace(0,1,zz.shape[0]),zz,label="solution initialisee")
    plt.legend()
    return(z,zz)
#z,zz=checkbaca(ldu=1,ab=2.0,ep=1.0,muu=1,mud=1,tau=300,T=50,nb=10)
#a la fin de la periode  cela ne correspond plus mais je ne vois pas pourquoi.




#############################################################################
###### jeudi 31 janvier 2019
########## generation des figures complexes pour le papier
#import seaborn as sns;
#sns.set()
#sns.set_color_codes()

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )

def trouvelsinmu(lz,of):
    if ((of>=1) or (of< 1-2*lz)):
        return((False,0.0,0.0))
    else:
        a=0.5 - (np.arcsin(((1-of)/lz) -1))/(2*np.pi)
        return((True,a,1.5-a))

def periodise(f):
    #retourne la fonction qui etait definie sur [0,1] periodisee sur [0,2]
    def g(t):
        return(f(t-np.floor(t)))
    return(g)
def periodiseetintegre(f):
    
    def g(t):
        k=np.floor(t)
        return(f(t-k)+k*f(1))
    return(g)

def tauxphietpe(gamma=0.3,lzero=3.0,lT=[0.2,1,5,20,50,100],offsetlam=0.0):
    def flechewinter(hauteur,xmin,xmax,couleur=cgris):
        hdelta=0.05
        #plt.arrow(xmax-hdelta,hauteur,-(xmax-xmin)+2*hdelta,0,shape='full',  length_includes_head=True,color=couleur,alpha=1.0,linewidth=15,head_starts_at_zero=True,head_width=5e-2)
        plt.arrow(xmax,hauteur,-(xmax-xmin)+2*hdelta,0,shape='full',  length_includes_head=True,
                  color=couleur,alpha=1.0,linewidth=15,head_starts_at_zero=True,head_width=5e-2)
    def grise():
        bandegris(1-gamma, 1.0)
        bandegris(1+1-gamma,2.0)
    #atracer,asin,bsin=bdp.trouvelsinmu(lzero*(1-gamma),0.0)
    def axsingrise(ax):
        if (atracer):
            axbandegris(ax,asin,bsin)
            axbandegris(ax,asin+1,bsin+1)
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def la(t):
        if (t< 1-gamma):
            return(lzero+offsetlam)
        else:
            return(0.0+offsetlam)
    def La(t):
        if (t< 1-gamma):
            return((lzero+offsetlam)*t)
        else:
            return(lzero*(1-gamma)+offsetlam*t)
    def phi(t):
        return(La(t) -Mu(t))
    #il faut ajuster le lzerosin pour que les processus aient le meme rzero

    def lasin(t):
        return(0.5*lzero*(1+np.sin(2*pi*t))+offsetlam)
    def Lasin(t):
        return(0.5*lzero*(t + (1/(2*pi))*(1-np.cos(2*pi*t)))+t*offsetlam)
    def phisin(t):
        return(Lasin(t) -Mu(t))

    print("phi(1),phisin(1)",phi(1),phisin(1))
    
    pla=periodise(la)
    pmu=periodise(mu)
    plasin=periodise(lasin)
    
    et=np.linspace(0,2,200)
    t=np.linspace(0,1,100)

    pphi=periodiseetintegre(phi)
    pphisin=periodiseetintegre(phisin)
    vpphi=np.array([pphi(s) for s in et])
    vpphisin=np.array([phisin(s) for s in et])
    vphimax=max(vpphi.max(),vpphisin.max())
    
    nblignes=3 
    f,ax=plt.subplots(nrows=nblignes, ncols=2, figsize=[18, 16]) #pour les graphiques beamer
    #f,ax=plt.subplots(nrows=3, ncols=2, figsize=[11, 16]) #pour les graphiques A4
    #plt.tight_layout(h_pad=3.5)
    plt.subplots_adjust(left=0.125,right=0.9,bottom=0.1,top=0.9,hspace=0.5)

    #on trace les taux de naissance et de mort step
    #plt.subplot(nblignes,2,1)
    axc=ax[0,0]
    axabsticks(axc)
    mettrelettreax(axc,'A',gauche=0.0)
    if (offsetlam<1):
        axbandegris(axc,1-gamma, 1.0)
        axbandegris(axc,1+1-gamma,2.0)
    axc.set_ylabel(r"\bf{Birth and Death rates}")
    axc.set_yticks((0,1,2,3))
    axc.set_yticklabels((r"\bf{0}",r"\bf{1}",r"\bf{2}",r"\bf{3}"))
    vpla=np.array([pla(s) for s in et])
    vplasin=np.array([plasin(s) for s in et])
    vmax=max(vpla.max(),vplasin.max())
    axc.plot(et,vpla,label=r"$\lambda(t)$",color='black')
    axc.plot(et,[pmu(s) for s in et],label=r"$\mu(t)$",color='black',dashes=[6,2])
    axc.axis([0.0, 2.0,0.0, 1.3*vmax]) #par defaut dans matplotlib sans seaborn
    axc.text(0.05,1.1,r"$\mu(t)$",fontsize=30)
    axc.text(0.05,la(0.0)+0.1,r"$\lambda(t)$",fontsize=30)

    #on trace les taux de naissance et de mort sinusoidaux
    #plt.subplot(nblignes,2,2)
    axc=ax[0,1]
    axabsticks(axc)
    mettrelettreax(axc,'B')
    axc.set_yticks((0,1,2,3))
    axc.set_yticklabels((r"\bf{0}",r"\bf{1}",r"\bf{2}",r"\bf{3}"))
    atracer,asin,bsin=trouvelsinmu(lzero*(1-gamma),offsetlam)
    axsingrise(axc)
    axc.plot(et,vplasin,label=r"$\lambda(t)$",color='black')
    axc.plot(et,[pmu(s) for s in et],label=r"$\mu(t)$",color='black',dashes=[6,2])
    axc.axis([0, 2.0, 0.0, 1.3*vmax]) #par defaut dans matplotlib sans seaborn
    
    ##########################################
    ##### courbe phi dans le cas step
    ##################################
    axc=ax[1,0]
    axabsticks(axc)
    mettrelettreax(axc,'C',gauche=0.0)

    if (offsetlam<1):
        axbandegris(axc,1-gamma, 1.0)
        axbandegris(axc,1+1-gamma,2.0)

    axc.set_ylabel(r"\bf{Integrated growth rate} $\varphi(t)$")
    axc.set_yticks((0.0,1.0,2.0))
    axc.set_yticklabels((r"\bf{0}",r"\bf{1}",r"\bf{2}"))
    axc.axis([0, 2.0, 0.0, 1.1*vphimax]) #par defaut dans matplotlib sans seaborn
    axc.plot(et,vpphi,label=r"$\varphi(t)$",color='black')

    tstar=(phi(1))/(lzero-1)
    minloc=phi(1)
    print("tstar=",tstar)
    axc.text(tstar-0.13,0.02,r"${t^*}$",fontsize=30)
    abst=np.linspace(tstar,1,100)

    line=axc.plot(abst,np.full_like(a=abst,fill_value=minloc),dashes=[6,2],color='black')[0]

    #add_arrow(line,position=0,direction='left',size=30,color='red')
    h=0.1
    axc.arrow(tstar+h,minloc,-h,0,shape='full', lw=0, length_includes_head=True, head_width=.05,color='black')

    ybst=np.linspace(0,minloc,100)

    axc.plot(np.full_like(a=ybst,fill_value=tstar),ybst,dashes=[6,2],color='black')
    axc.arrow(tstar,h,0,-h,shape='full', lw=0, length_includes_head=True, head_width=.05,color='black')
    axbandegris(axc,tstar,1-gamma,alpha=0.5)
    axbandegris(axc,1+tstar,1+1-gamma,alpha=0.5)

    ################################################################
    # on trace la courbe phi dans le cas sinusoidal
    axc=ax[1,1]
    axabsticks(axc)
    mettrelettreax(axc,'D')
    axc.set_yticks((0.0,1.0,2.0))
    axc.set_yticklabels((r"\bf{0}",r"\bf{1}",r"\bf{2}"))
    axc.axis([0, 2.0, 0.0, 1.1*vphimax]) #par defaut dans matplotlib sans seaborn

    axc.plot(et,vpphisin,label=r"$\varphi(t)$",color='black')
    indl=np.linspace(0.5,1,1000)
    x=np.array([phisin(s) for s in indl])
    w=np.where(x==x.min())
    tminloc=indl[w[0][0]]
    print("tminloc=",tminloc)
    minloc=phisin(tminloc)
    #maintenant on recherche tstar
    indt=np.linspace(0.0,0.5,1000)
    x=np.array([phisin(s) for s in indt])
    w=np.where(np.abs(x-minloc)<= 1e-3)
    tstarsin=indt[w[0][0]]
    print("tstarsin=",tstarsin,"phisin(tstar)",phisin(tstarsin),"phisin(tminloc)",phisin(tminloc))
    axc.text(tstarsin-0.13,0.02,r"${t^*}$",fontsize=30)
    abst=np.linspace(tstarsin,tminloc,100)

    axc.plot(abst,np.full_like(a=abst,fill_value=minloc),dashes=[6,2],color='black')

    #la fleche pointillee depuis le minimum local jusque a tstar
    h=0.1
    axc.arrow(tstarsin+h,minloc,-h,0,shape='full', lw=0, length_includes_head=True, head_width=.05,color='black')

    ybst=np.linspace(0,minloc,100)

    axc.plot(np.full_like(a=ybst,fill_value=tstarsin),ybst,dashes=[6,2],color='black')
    axc.arrow(tstarsin,h,0,-h,shape='full', lw=0, length_includes_head=True, head_width=.05,color='black')

    def axsinwicetwinter(ax):
        axsingrise(axc)
        axbandegris(axc,tstarsin,asin,alpha=0.5)
        axbandegris(axc,1+tstarsin,1+asin,alpha=0.5)
        return()
    axsinwicetwinter(axc)

    #### proba emergence cas sinusoidal
    #plt.subplot(nblignes,2,6)
    axc=ax[2,1]
    mettrelettreax(axc,'F')
    axabsticks(axc)
    axc.set_yticks((0.0,0.2,0.4,0.6))
    axc.set_yticklabels((r"\bf{0}",r"\bf{0.2}",r"\bf{0.4}",r"\bf{0.6}"))
    axc.set_xlabel(r"\bf{Time of pathogen introduction} ${t_0}$",fontsize=22)
    vpemax=0.0
    cs=CubicSpline(np.array([0.0,10,100]),np.array([0.2,0.5,0.9]))

    for T in lT:
        pem=nncalculepe(lasin,Lasin,mu,Mu,T)
        ppem=periodise(pem)
        valppem=np.array([ppem(s) for s in et])
        axc.plot(et,valppem,label=r"T="+str(T),color=plt.cm.YlOrRd(cs(T)))
        vpemax=max(vpemax,valppem.max())

    axc.axis([0, 2.0, 0.0, 1.1*vpemax]) #par defaut dans matplotlib sans seaborn
    def singuess(t):
        if (lasin(t)>=mu(t)):
            return(1-(mu(t)/lasin(t)))
        else:
            return(0.0)
    
    psing=periodise(singuess)
    axsinwicetwinter(axc)
    axc.plot(et,[psing(s) for s in et],label="guess(t)",color='black',dashes=[6,2])


    #### proba emergence cas step
    #axc.subplot(nblignes,2,5)
    axc=ax[2,0]
    mettrelettreax(axc,'E',gauche=0.0)
    axabsticks(axc)
    axc.set_yticks((0.0,0.2,0.4,0.6))
    axc.set_yticklabels((r"\bf{0}",r"\bf{0.2}",r"\bf{0.4}",r"\bf{0.6}"))
    if (offsetlam<1):
        axbandegris(axc,1-gamma, 1.0)
        axbandegris(axc,1+1-gamma,2.0)
        
    axc.set_ylabel(r"\bf{Emergence probability} ${p_e(t_0T,T)}$")        
    axc.axis([0, 2.0, 0.0, 1.1*vpemax]) #par defaut dans matplotlib sans seaborn
    axc.set_xlabel(r"\bf{Time of pathogen introduction} ${t_0}$",fontsize=22)
    #plt.xlabel("Introduction time of the infected")

    for T in lT:
        pem=nncalculepe(la,La,mu,Mu,T)
        ppem=periodise(pem)
        color=plt.cm.YlOrRd(cs(T))
        axc.plot(et,[ppem(s) for s in et],label=r"T="+str(T),color=color)
    
    def stepguess(t):
        if (la(t)>=mu(t)):
            return(1-(mu(t)/la(t)))
        else:
            return(0.0)
    
    psg=periodise(stepguess)
    #plt.plot(et,[psg(s) for s in et],label=r"guess($t_0$)",color='black',dashes=[6,2])
    axc.plot(et,[psg(s) for s in et],color='black',dashes=[6,2])
    #flechewinter(0.6,bdp.trouvepremierzero(psg,0.1,0.9),1.0)
    tstarstep=phi(1)/(lzero-1)
    astep,bstep=trouvepremieretdernierzero(ppem,0.1,1.0)
    print("tstarstep=",tstarstep)
    axbandegris(axc,tstarstep,1-gamma,alpha=0.5)
    axbandegris(axc,1+tstarstep,1+1-gamma,alpha=0.5)
    #bandegris(astep,bstep,alpha=0.5)
    #bandegris(1+astep,1+bstep,alpha=0.5)

    legend = axc.legend(fontsize=12,loc='lower left')
    frame = legend.get_frame()
    frame.set_linewidth(0.0)
    
    plt.savefig("ratesandemergenceproba.pdf",dpi=300,pad_inches=0.1)
        


#########################################################################
####  vendredi premier février 2019 : attaquons nous a la seconde figure

#tout d'abord on reprend notre calcul de la strategie de controle optimale
    
def nsimusin(lzero=4.0,C=0.2,voir=False):
    #ne pas ajuster le lzero : cela doit etre fait dans la fonction qui appelle celle ci
    
    def la(t):
        return(lzero*(1+np.sin(2*pi*t)))
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def La(t):
        return(lzero*(t + (1/(2*pi))*(1-np.cos(2*pi*t))))
    pla=periodise(la)
    pmu=periodise(mu)
    pLa=periodiseetintegre(La)
    pMu=periodiseetintegre(Mu)
    
    def sinqextmoy(tun,rhom,C=C):
        #on peut prendre tdeux >=1, mais il faut imposer rhom >= C
        #et prendre les fonctions periodisees
        assert(rhom >= C)
        deltat=C/rhom
        tdeux=tun+deltat
        if (tdeux <=1):
            def rho(t):
                return(rhom if ((tun<t) and (t< tdeux)) else 0.0)
            def Larho(t):
                return(La(t) - rhom*(La(min(t,tdeux)) -La(min(t,tun))))

        else:
            def rho(t):
                return(rhom if ((tun<t) or (t< tdeux-1)) else 0.0)
            def Larho(t):
                return(La(t) - rhom*(La(min(t,tdeux-1)) + La(t) -La(min(t,tun))))
           
        prho=periodise(rho)
            
        def plarho(t):
            return(pla(t)*(1-prho(t)))
        pLarho=periodiseetintegre(Larho)      

        v=npeinfmoy(plarho,pLarho,pmu,pMu,voir=voir)
        return(v)
    return(sinqextmoy)
                   
def ngrillesimusin(nbpt,lzero=4.0,C=0.2,tunmax=0.5):
    #print("ngrillesimusin,lzero=",lzero,",C=",C)
    f=nsimusin(C=C,lzero=lzero)
    tun=np.arange(0,tunmax,tunmax/nbpt)
    rho=(1.0/nbpt)+np.arange(C,1.0,1.0/nbpt)
    zs = np.array([[f(i,j) for j in rho] for i in tun])
    return(zs)

def ntestsimusin(nbpt=100,C=0.3,lzero=4.0,tunmax=0.5):
    z=ngrillesimusin(nbpt=nbpt,C=C,lzero=lzero,tunmax=tunmax)
    np.save("ntestsimusinN"+str(nbpt)+"C"+str(C)+"lzero"+str(lzero),z)
    return(z)

def nsinvoir(z,C=0.2,tunmax=0.5):
    N=(z.shape)[0]
    tun=np.arange(0.0,tunmax,tunmax/N)
    rho=(1.0/N)+np.arange(C,1.0,1.0/N)
    
    X,Y=np.meshgrid(rho,tun)
    fig, ax = plt.subplots()
    CS = ax.contour(X,Y,z,cmap='flag')
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_ylabel(r"$t_1$")
    ax.set_xlabel(r"$\rho_M$")
    m=z.min()
    y=np.where(z==m)
    #xopt=(y[0][0]/N)*tunmax
    xopt=tun[y[0][0]]
    #yopt=(y[1][0]/N)*(1-C)+C
    yopt=rho[y[1][0]]
    titre=r"Mean emergence probability: Contour Plot. Cost="+str(C)+"\n"+ r""" minimal $<p_e>$=""" +'{:.4f}'.format(m)+r""" for $t_1$=""" + str(xopt)+r""", $\rho_M=$"""+str(yopt)
    ax.set_title(titre)
    plt.show()
def nnsinvoir(z,C=0.2,tunmax=0.5):
    N=(z.shape)[0]
    tun=np.arange(0.0,tunmax,tunmax/N)
    rho=(1.0/N)+np.arange(C,1.0,1.0/N)
    
    X,Y=np.meshgrid(rho,tun)
    fig, ax = plt.subplots()
    zmin=z.min()
    zmax=z.max()
    c = ax.pcolormesh(X,Y,z,cmap='RdBu',vmin=zmin,vmax=zmax)
    ax.set_ylabel(r"$t_1$")
    ax.set_xlabel(r"$\rho_M$")
    m=z.min()
    y=np.where(z==m)
    #xopt=(y[0][0]/N)*tunmax
    xopt=tun[y[0][0]]
    #yopt=(y[1][0]/N)*(1-C)+C
    yopt=rho[y[1][0]]
    titre=r"Mean emergence probability: Cost="+str(C)+"\n"+ r""" minimal $<p_e>$=""" +'{:.4f}'.format(m)+r""" for $t_1$=""" + str(xopt)+r""", $\rho_M=$"""+'{:.4f}'.format(yopt)
    ax.set_title(titre)
    fig.colorbar(c,ax=ax)
    plt.show()

#jeudi 7 fevrier : il faut ecrier une autre fonction peifmoy, qui tienne compte que l'on peut avoir tdeux plus grand que 1


def npeinfmoy(la,La,mu,Mu,N=200,voir=False):
    def phi(t):
        return(La(t)-Mu(t)) #les fonctions en argument sont deja periodisees
    
    lesphi=np.array([phi(i/N) for i  in np.arange(2*N+1)])
    def peinf(i):
        t=i/N
        #print(i,t,la(t),mu(t),La(t),Mu(t))
        if (la(t) <= mu(t)):
            return(0.0)
        z=lesphi[i+1:i+N]- lesphi[i] #on regarde la courbe de taux sur une periode apres i
        if (z.min()<=0): #s 'il y a un piege, la proba d'extinctinest nulle
            return(0.0) 
        else:
            return(1-(mu(t)/la(t)))
    ab=np.arange(N+1)
    val=np.array([peinf(i) for i in ab])
    return(val.mean())



def bacanpeinfmoy(la,La,mu,Mu,N=200,voir=False,nbtau=10):
    T=40 #plus grand cela fait exploser l'integration numerique
    pe=nncalculepe(la,La,mu,Mu,T,limit=100)
    ppe=periodise(pe)
    def phi(t):
        return(La(t)-Mu(t)) #les fonctions en argument sont deja periodisees
    
    lesphi=np.array([phi(i/N) for i  in np.arange(2*N+1)])
    def peinf(i):
        t=i/N
        #print(i,t,la(t),mu(t),La(t),Mu(t))
        if (la(t) <= mu(t)):
            return(0.0)
        z=lesphi[i+1:i+N]- lesphi[i] #on regarde la courbe de taux sur une periode apres i
        #print("z=",z)
        if (z.min()<=0): #s 'il y a un piege, la proba d'extinctinest nulle
            return(0.0) 
        else:
            return(1-(mu(t)/la(t)))
    ab=np.arange(N+1)
    val=np.array([peinf(i) for i in ab])
    def pf(t):
        "on fabrique la fonction a partir des valeurs entieres"
        return(val[int(N*t)])
    ppf=periodise(pf)

    Lambda=[[la]]
    Mu=[mu]
    z=vfbaca(Lambda,Mu,T=T,tv=False,nbtau=nbtau)
    z=z.reshape(len(z),)
    def pz(t):
        return(z[int(len(z)*t)])
    ppz=periodise(pz)

    
    if voir:
        abcisse=np.arange(0.0,2.0,1.0/(2*N))
        ordonnee=[ppf(t) for t in abcisse]
        #print("ordonnee=",ordonnee)
        plt.plot(abcisse,ordonnee,label=r"$p_{e,\infty}$")
        plt.plot(abcisse,[ppe(t) for t in abcisse],label=r"$p_{e,T}(tT)$")
        plt.plot(abcisse,[ppz(t) for t in abcisse],label=r"$ppz(t)$")
        plt.legend()
    #on fait une integration par la methode des trapezes
    trapint=np.trapz(val,ab/N)
    directint=quad(ppf,0,1)[0]
    pemoy=(quad(pe,0,1,limit=100)[0])
    pzmoy=(quad(pz,0,1,limit=100)[0])
    #print("trapint=",trapint,"directint=",directint,"moyenne de val",val.mean())
    print("trapint=",trapint,"directint=",directint,"moyenne de val",val.mean(),"pemoy=",pemoy,"pzmoy=",pzmoy,"z.mean",z.mean())
    return(np.trapz(val,ab/N))


####################################################################################
########### vendredi 8 fevrier 2019
########### enfin je m'attaque a la figure 2

def controltauxphietpe(gamma=0.3,lzero=3.0,tunmax=1.0,T=70,nbpt=100,C=0.2,beamer=False,tunstepopt=0.2,rhomstepopt=0.85):
    r"on fixe a priori une grande periode"
    #couleurcontrole='lightblue'
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def la(t):
        if (t< 1-gamma):
            return(lzero)
        else:
            return(0.0)
    def La(t):
        if (t< 1-gamma):
            return(lzero*t)
        else:
            return(lzero*(1-gamma))
    def phi(t):
        return(La(t) -Mu(t))
    #il faut ajuster le lzerosin pour que les processus aient le meme rzero

    def lasin(t):
        return(lzero*(1-gamma)*(1+np.sin(2*pi*t)))
    def Lasin(t):
        return(lzero*(1-gamma)*(t + (1/(2*pi))*(1-np.cos(2*pi*t))))
    def phisin(t):
        return(Lasin(t) -Mu(t))

    def grise():
        bandegris(1-gamma, 1.0)
        bandegris(1+1-gamma,2.0)
    def bandegrisbarre(a,b):
        plt.axvspan(a, b, color='lightgray', alpha=0.2, lw=0,hatch='///')
    def grisebarre():
        bandegrisbarre(1-gamma, 1.0)
        bandegrisbarre(1+1-gamma,2.0)
    def flechewinter(hauteur,xmin,xmax,couleur='silver'):
        hdelta=0.05
        plt.arrow(xmax-hdelta,hauteur,-(xmax-xmin)+2*hdelta,0,shape='full',  length_includes_head=True,color=couleur,linewidth=10,head_starts_at_zero=True)


    plt.rc('lines', linewidth=3.5, color='b')
    
    atracer,asin,bsin=trouvelsinmu(lzero*(1-gamma),0.0)
    def singrise():
        if (atracer):
            bandegris(asin,bsin)
            bandegris(asin+1,bsin+1)

    #print("phi(1),phisin(1)",phi(1),phisin(1))
    #on recupere les valeurs de tun et rhom qui donnent le minimum
    #tun,rhom=minopt(nbpt,C,lzero,tunmax=1.0)
    lzeromodif=round(lzero*(1-gamma)*100)/100

    tun,rhom=minopt(nbpt,C,lzeromodif,tunmax=1.0,prefixe="unif")
   
    #on definit les fonctions corespondant au controle optimal pour la sinusoide
    deltat=C/rhom
    tdeux=tun+deltat
    print("optimum sinus:tun,tdeux,rhom=",tun,tdeux,rhom)
    if (tdeux <=1):
        def rhosinopt(t):
            return(rhom if ((tun<t) and (t< tdeux)) else 0.0)
        def Larhosinopt(t):
            return(Lasin(t) - rhom*(Lasin(min(t,tdeux)) -Lasin(min(t,tun))))

    else:
        def rhosinopt(t):
            return(rhom if ((tun<t) or (t< tdeux-1)) else 0.0)
        def Larhosinopt(t):
            return(Lasin(t) - rhom*(Lasin(min(t,tdeux-1)) + Lasin(t) -Lasin(min(t,tun))))
    pla=periodise(la)
    pmu=periodise(mu)
    plasin=periodise(lasin)
    

    prhosinopt=periodise(rhosinopt)

    def plarhosinopt(t):
        return(plasin(t)*(1-prhosinopt(t)))
    pLarhosinopt=periodiseetintegre(Larhosinopt)      

    def phirhosinopt(t):
        return(Larhosinopt(t)-Mu(t))

    ############################## on definit la fonction optimale pour le cas step
    #tdso=1-gamma-C*lzero/(lzero-1)
    #rhoms=C/(1-gamma -tdso)
    #print("tdso=",tdso)

    #attention , changement de strategie, on fixe la stratégie optimale
    # dans les parametres
    tunso,rhoms=tunstepopt,rhomstepopt
    tdso=tunso +(C/rhoms)
    assert(tdso<=1-gamma)
    def lasopt(t):
        if (t <=tunso):
            return(lzero)
        elif (t<= tdso):
            return(lzero*(1-rhoms))
        elif (t<=1-gamma):
            return(lzero)
        else:
            return(0.0)
    plasopt=periodise(lasopt)

    def Lasopt(t):
        if (t <=tunso):
            return(t*lzero)
        elif (t<= tdso):
            return(tunso*lzero+(t-tunso)*lzero*(1-rhoms))
        elif (t<= 1-gamma):
            return(tunso*lzero+(tdso-tunso)*lzero*(1-rhoms) + (t-tdso)*lzero)
        else:
            return(tunso*lzero+(tdso-tunso)*lzero*(1-rhoms) + (1-gamma-tdso)*lzero)
    def phisopt(t):
        return(Lasopt(t)-Mu(t))

    ############# la on va tracer les courbes    
    et=np.linspace(0,2,200)
    t=np.linspace(0,1,100)
    #plt.rcParams['figure.figsize'] = [16, 10]
    if (beamer):
        f,ax=plt.subplots(nrows=3, ncols=2, figsize=[16, 9])
    else:
        f,ax=plt.subplots(nrows=3, ncols=2, figsize=[9, 16])
    
    plt.subplot(3,2,1)
    grise()
    plt.ylabel("Birth and Death rates")
    vpla=np.array([pla(s) for s in et])
    vplasopt=np.array([plasopt(s) for s in et])
    vplasin=np.array([plasin(s) for s in et])
    vplasinopt=np.array([plarhosinopt(s) for s in et])
    vmax=max(vpla.max(),vplasin.max(),vplasinopt.max())
    plt.plot(et,vpla,label=r"$\lambda(t)$",color='black')
    plt.plot(et,vplasopt,label=r"$\lambda_\rho(t)$",color=couleurcontrole)
    plt.plot(et,[pmu(s) for s in et],label=r"$\mu(t)$",color='black',dashes=[6, 2])
    plt.axis([0.0, 2.0,0.0, 1.1*vmax]) #par defaut dans matplotlib sans seaborn
    #plt.legend(fontsize=12)

    plt.subplot(3,2,2)
    singrise()
    plt.plot(et,vplasin,label=r"$\lambda(t)$",color='black')
    plt.plot(et,vplasinopt,label=r"$\lambda_\rho(t)$",color=couleurcontrole)
    plt.plot(et,[pmu(s) for s in et],label=r"$\mu(t)$",color='black',dashes=[6, 2])
    plt.axis([0, 2.0, 0.0, 1.1*vmax]) 
    #plt.legend()

    ##########################################
    ##### courbe phi dans le cas step
    ##################################
    plt.subplot(3,2,3)
    grise()
    pphi=periodiseetintegre(phi)
    pphisopt=periodiseetintegre(phisopt)
    pphisin=periodiseetintegre(phisin)
    pphirhosinopt=periodiseetintegre(phirhosinopt)
    vpphi=np.array([pphi(s) for s in et])
    vpphisopt=np.array([pphisopt(s) for s in et])
    vpphisin=np.array([pphisin(s) for s in et])
    vpphirhosinopt=np.array([pphirhosinopt(s) for s in et])
    vphimax=max(vpphi.max(),vpphisin.max(),vpphirhosinopt.max(),vpphisopt.max())
    vphimin=vpphirhosinopt.min()
    plt.ylabel(r"Integrated growth rate")
    plt.axis([0, 2.0, 1.1*vphimin, 1.1*vphimax]) #par defaut dans matplotlib sans seaborn
    plt.plot(et,vpphi,label=r"$\varphi(t)$",color='black')
    plt.plot(et,vpphisopt,label=r"$\varphi_\rho(t)$",color=couleurcontrole)
    tracetstar(pphi,0.75,1.25,0.0,0.75,vphimin)
    #tracetstar(pphisopt,tdso,1+tdso,0.0,tdso,vphimin)
    tracetstar(pphisopt,0.2,0.5,0.0,0.1,vphimin)
    tracetstar(pphisopt,0.75,1,0.3,0.7,vphimin)
    #plt.legend(fontsize=12)

    ################################################################
    # on trace la courbe phi dans le cas sinusoidal
    plt.subplot(3,2,4)
    singrise()
    plt.axis([0, 2.0, 1.1*vphimin, 1.1*vphimax]) #par defaut dans matplotlib sans seaborn
    
    plt.plot(et,vpphisin,label=r"$\varphi(t)$",color='black')
    plt.plot(et,vpphirhosinopt,label=r"$\varphi_\rho(t)$",color=couleurcontrole)

    tracetstar(pphisin,0.5,1.0,0.0,0.5,vphimin)
    tracetstar(pphirhosinopt,1.0,1.5,0.1,0.7,vphimin)
    #plt.legend()
    #### proba emergence cas sinusoidal
    plt.subplot(3,2,6)
    singrise()
    plt.xlabel("Introduction time of the infected")

    #plt.ylabel(r"Emergence probability $p_e(t_0T,T)$, T="+str(T))
    pem=nncalculepe(lasin,Lasin,mu,Mu,T)
    ppem=periodise(pem)
    valppem=np.array([ppem(s) for s in et])
    plt.plot(et,valppem,label=r"$p_e$",color='black')
    vpemax=valppem.max()
    plt.axis([0, 2.0, 0.0, 1.1*vpemax]) #par defaut dans matplotlib sans seaborn
    #mainenant pour le controle optimal
    pem=nncalculepe(plarhosinopt,Larhosinopt,mu,Mu,T)
    ppem=periodise(pem)
    valppem=np.array([ppem(s) for s in et])
    plt.plot(et,valppem,label=r"$p_{e,opt}$",color=couleurcontrole)
    flechewinter(0.4,trouvepremierzero(ppem,0.4,0.7),trouvemax(ppem,1.0,1.5),couleur=couleurcontrole)

    #### proba emergence cas step
    plt.subplot(3,2,5)
    grise()
    plt.ylabel(r"$p_e(t_0T,T)$, T="+str(T))        
    plt.axis([0, 2.0, 0.0, 1.1*vpemax]) #par defaut dans matplotlib sans seaborn
    plt.xlabel("Introduction time of the infected")

    pem=nncalculepe(la,La,mu,Mu,T)
    ppem=periodise(pem)
    plt.plot(et,[ppem(s) for s in et],label=r"$p_e$",color='black')

    flechewinter(0.1,trouvepremierzero(pem,0.1,0.9),1.0,couleur='lightgrey')
    pemsopt=nncalculepe(lasopt,Lasopt,mu,Mu,T)
    ppemsopt=periodise(pemsopt)
    plt.plot(et,[ppemsopt(s) for s in et],label=r"$p_{e,opt}$",color=couleurcontrole)
    flechewinter(0.4,trouvepremierzero(pemsopt,0.1,0.9),1.0,couleur=couleurcontrole)
    #plt.legend(fontsize=12)

    plt.savefig("controlratesandemergenceproba.pdf",dpi=300)


def trouvepremierzero(f,a,b,voir=False,tolerance=1e-3):
    N=100
    t=np.linspace(a,b,N)
    z=np.array([f(s) for s in t])
    if voir:
        plt.plot(t,z)
    #w=np.where(z==z.min())
    w=np.where(z<=tolerance)
    return(a+w[0][0]*(b-a)/N)

def trouvepremieretdernierzero(f,a,b,voir=False,tolerance=1e-3):
    N=100
    t=np.linspace(a,b,N)
    z=np.array([f(s) for s in t])
    if voir:
        plt.plot(t,z)
    #w=np.where(z==z.min())
    w=np.where(z<=tolerance)
    assert(len(w[0])!=0)
    return((a+w[0][0]*(b-a)/N),(a+w[0][-1]*(b-a)/N))









def testtpz():
    def lam(x):
        return(1.5*(1 +np.sin(2*pi*x)))
    def mu(x):
        return(1.0)
    def f(x):
        return(lam(x)-mu(x) + np.abs(lam(x)-mu(x)))
    return(trouvepremierzero(f,0.2,1.0,voir=True),trouvemax(f,0.1,1.0))

def trouvemax(f,a,b):
    N=100
    t=np.linspace(a,b,N)
    z=np.array([f(s) for s in t])
    w=np.where(z==z.max())
    return(a+w[0][0]*(b-a)/N)
   

def minopt(nbpt,C,lzero,tunmax=1.0,prefixe=""):
    nomfic=prefixe+"ntestsimusinN"+str(nbpt)+"C"+str(C)+"lzero"+str(lzero)+".npy"
    print("minopt: nomfic",nomfic)
    zz=np.load(nomfic)
    N,M=zz.shape
    tun=np.arange(0.0,tunmax,tunmax/N)
    rho=(1.0/N)+np.arange(C,1.0,1.0/N)
    y=np.where(zz==zz.min())
    xopt=tun[y[0][0]]
    yopt=rho[y[1][0]]
    return(xopt,yopt)


def tracetstar(fonctaux,mintloc,maxtloc,mintstar,maxtstar,minfonctaux,couleur='black',vert=True):
    r"""on trouve numeriquement le tstar pour la courbe phi periodisee fonctaux. le minimumlocal
est entre mintloc et maxtloc

minfonctaux ne sert a rien"""
    indl=np.linspace(mintloc,maxtloc,1000)
    x=np.array([fonctaux(s) for s in indl])
    w=np.where(x==x.min())
    tminloc=indl[w[0][0]]
    minloc=fonctaux(tminloc)
    #maintenant on recherche tstar
    indt=np.linspace(mintstar,maxtstar,1000)
    x=np.array([fonctaux(s) for s in indt])
    w=np.where(np.abs(x-minloc)<= 1e-3)
    tstar=indt[w[0][0]]
    #print("tstar=",tstar,"phisin(tstar)",phisin(tstar),"phisin(tminloc)",phisin(tminloc))
    #plt.text(tstar+0.05,0,r"$t^*$")
    #on trace les pointilles horizontaux
    abst=np.linspace(tstar,tminloc,100)
    
    plt.plot(abst,np.full_like(a=abst,fill_value=minloc),dashes=[6,2],color=couleur)

    #la fleche pointillee depuis le minimum local jusque a tstar
    h=0.1
    plt.arrow(tstar+h,minloc,-h,0,shape='full', lw=0, length_includes_head=True, head_width=.05,color=couleur)

    if vert:
        #puis on trace les pointilles verticaux
        ybst=np.linspace(minfonctaux,minloc,100)
    
        plt.plot(np.full_like(a=ybst,fill_value=tstar),ybst,dashes=[6,2],color=couleur)
        plt.arrow(tstar,minfonctaux+h,0,-h,shape='full', lw=0, length_includes_head=True, head_width=.05,color=couleur)
      


###############################################################
######### lundi 11 fevrier 2019
### influence de la densite d'introduction

# d'abord le calcul du rzero optimal dans le cas sinusoidal

def ngrillerzerosin(nbpt,C=0.2,lzero=2.1,tunmax=1.0):
    f=rzerosin(C=C,lzero=lzero)
    tun=np.arange(0,tunmax,tunmax/nbpt)
    rho=(1.0/nbpt)+np.arange(C,1.0,1.0/nbpt)
    zs = np.array([[f(i,j) for j in rho] for i in tun])
    return(zs)

def ntestrzerosin(nbpt=100,C=0.2,lzero=2.1):
    z=ngrillerzerosin(nbpt=nbpt,C=C,lzero=lzero)
    np.save("ntestrzerosinN"+str(nbpt)+"C"+str(C)+"lzero"+str(lzero),z)
    return(z)

def nrzerosinvoir(z,C=0.2,tunmax=1.0):
    N=(z.shape)[0]
    tun=np.arange(0.0,tunmax,tunmax/N)
    rho=(1.0/N)+np.arange(C,1.0,1.0/N)
    
    X,Y=np.meshgrid(rho,tun)
    fig, ax = plt.subplots()
    zmin=z.min()
    zmax=z.max()
    ax.set_ylabel(r"$t_1$")
    ax.set_xlabel(r"$\rho_M$")
    m=z.min()
    y=np.where(z==m)
    xopt=tun[y[0][0]]
    yopt=rho[y[1][0]]
    titre=r"Basic Reproduction Number: Contour Plot. Cost="+str(C)+"\n"+ r""" minimal $R_0$=""" +'{:.4f}'.format(m)+r""" for $t_1$=""" + str(xopt)+r""", $\rho_M=$"""+str(yopt)
    ax.set_title(titre)
    c = ax.pcolormesh(X,Y,z,cmap='RdBu',vmin=zmin,vmax=zmax)
    fig.colorbar(c,ax=ax)
    plt.show()


    
def nrzerosin(lzero,C):
    def la(t):
        return(lzero*(1+np.sin(2*pi*t)))
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def La(t):
        return(lzero*(t + (1/(2*pi))*(1-np.cos(2*pi*t))))
    
    def rzerorho(tun,rhom,C=C):
        deltat=min(C/rhom,1-tun)
        tdeux=tun+deltat
        return((La(1) -rhom*(La(tdeux)-La(tun)))/Mu(1))

    return(rzerorho)

# maintenant on fait la heatmap pour optimiser pe en rajoutant le rzero optimal
def nnnsinvoir(zsin,rzeropt,tunrzeropt,rhomrzeropt,C=0.2,tunmax=1.0,beamer=False):
    N=(zsin.shape)[0]
    tun=np.arange(0.0,tunmax,tunmax/N)
    rho=(1.0/N)+np.arange(C,1.0,1.0/N)
    
    X,Y=np.meshgrid(rho,tun)
    if (beamer):
        fig, ax = plt.subplots(figsize=[16, 12])
    else:
        fig, ax = plt.subplots(figsize=[12, 16])
    ax.text(-0.1, 1.1, r"\textbf{B}", transform=ax.transAxes, 
            fontsize=60)    
    plt.tight_layout()
        
    zsinmin=zsin.min()
    zsinmax=zsin.max()
    c = ax.pcolormesh(X,Y,zsin,cmap=Lacolormap,vmin=zsinmin,vmax=zsinmax)

    ax.set_ylabel(r"\bf{Start of control} ${t_1}$",fontsize=40)
    ax.set_xlabel(r"\bf{Intensity of control} ${\rho_M}$",fontsize=40,labelpad=10)
    ax.set_yticks((0.0,0.2,0.4,0.6,0.8))
    ax.set_yticklabels((r"\bf{0}",r"\bf{0.2}",r"\bf{0.4}",r"\bf{0.6}",r"\bf{0.8}"))
    ax.set_xticks((0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0))
    ax.set_xticklabels((r"\bf{0.2}",r"\bf{0.3}",r"\bf{0.4}",r"\bf{0.5}",r"\bf{0.6}",r"\bf{0.7}",r"\bf{0.8}",r"\bf{0.9}",r"\bf{1.0}"))
    ax.axis([C+1/N,1.1,0,1.0])
    m=zsin.min()
    y=np.where(zsin==m)
    #xopt=(y[0][0]/N)*tunmax
    xopt=tun[y[0][0]]
    #yopt=(y[1][0]/N)*(1-C)+C
    yopt=rho[y[1][0]]
    titre=r"Control strategies for sinusoidal birth rate: Cost="+str(C)+"\n"+ r""" minimal $<p_e>$=""" +'{:.4f}'.format(m)+r""" for $t_1$=""" + str(xopt)+r""", $\rho_M=$"""+'{:.4f}'.format(yopt)+"\n"+r"minimal $R_0$="+'{:.4f}'.format(rzeropt) +r""" for $t_1$=""" + str(tunrzeropt)+r""", $\rho_M=$"""+'{:.4f}'.format(rhomrzeropt)
    #ax.set_title(titre) #a remettre lorsque l'on veut lire les valeurs mai 2019)
    #fig.colorbar(c,ax=ax) #pas de barre de couleurs le 29 mai 2019
    #ax.plot(rhomrzeropt,tunrzeropt,marker='x',color=Lacolormarker,markersize=20)
    ax.plot(rhomrzeropt,tunrzeropt,marker='x',color='black',markersize=30)
    ax.plot(yopt,xopt,marker='x',color=couleurcontrole,markersize=30)
    #cercle=plt.Circle((yopt,xopt),0.01,color=couleurcontrole,fill=True)
    #ax.add_artist(cercle)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.text(rhomrzeropt-0.05,tunrzeropt+0.05,r"${R_{0,opt}}$",color='black',fontsize=40)
    plt.savefig("optimalpeandrzero.pdf",dpi=300,bbox_inches='tight',pad_inches=0.2)
    plt.show()



#on rajoute en parametre la fonction densite d'introduction pour le calcul de la proba d'emergence

def demicercledensity(t):
    assert((t>=0) and (t<=1))
    return((8/np.pi)*np.sqrt(t-t*t))
def unifdensity(t):
    return(1.0)
def betadensity(t):
    return(beta.pdf(t,2.0,5.0))
def sinusdensity(t):
    return(1+np.sin(2*np.pi*t))

#pour voir la densité du demicercle
def voirdemicercle():
    t=np.linspace(0,1,100)
    plt.plot(t,[demicercledensity(s) for s in t])
    print("surface=",(quad(demicercledensity,0,1))[0])

#pour voir la densité beta
def voirbeta(a=4,b=5):
    t=np.linspace(0,1,100)
    plt.plot(t,[beta.pdf(s,a,b) for s in t])
   

def nnsimusin(intdens,lzero=4.0,C=0.2,voir=False):
    #ne pas ajuster le lzero : cela doit etre fait dans la fonction qui appelle celle ci
    
    def la(t):
        return(lzero*(1+np.sin(2*pi*t)))
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def La(t):
        return(lzero*(t + (1/(2*pi))*(1-np.cos(2*pi*t))))
    pla=periodise(la)
    pmu=periodise(mu)
    pLa=periodiseetintegre(La)
    pMu=periodiseetintegre(Mu)
    
    def sinqextmoy(tun,rhom,C=C):
        #on peut prendre tdeux >=1, mais il faut imposer rhom >= C
        #et prendre les fonctions periodisees
        assert(rhom >= C)
        deltat=C/rhom
        tdeux=tun+deltat
        if (tdeux <=1):
            def rho(t):
                return(rhom if ((tun<t) and (t< tdeux)) else 0.0)
            def Larho(t):
                return(La(t) - rhom*(La(min(t,tdeux)) -La(min(t,tun))))

        else:
            def rho(t):
                return(rhom if ((tun<t) or (t< tdeux-1)) else 0.0)
            def Larho(t):
                return(La(t) - rhom*(La(min(t,tdeux-1)) + La(t) -La(min(t,tun))))
           
        prho=periodise(rho)
            
        def plarho(t):
            return(pla(t)*(1-prho(t)))
        pLarho=periodiseetintegre(Larho)      

        v,fonc=nnpeinfmoy(intdens,plarho,pLarho,pmu,pMu,voir=voir)
        return(v)
    return(sinqextmoy)
                   
def nngrillesimusin(intdens,nbpt,lzero=4.0,C=0.2,tunmax=0.5):
    #print("ngrillesimusin,lzero=",lzero,",C=",C)
    f=nnsimusin(intdens=intdens,C=C,lzero=lzero)
    tun=np.arange(0,tunmax,tunmax/nbpt)
    rho=(1.0/nbpt)+np.arange(C,1.0,1.0/nbpt)
    zs = np.array([[f(i,j) for j in rho] for i in tun])
    return(zs)

def nntestsimusin(prefixe="unif",nbpt=100,C=0.3,lzero=4.0,tunmax=0.5):
    if (prefixe=="unif"):
        z=nngrillesimusin(unifdensity,nbpt=nbpt,C=C,lzero=lzero,tunmax=tunmax)
    elif (prefixe=="demicercle"):
        z=nngrillesimusin(demicercledensity,nbpt=nbpt,C=C,lzero=lzero,tunmax=tunmax)
    elif (prefixe=="beta"):
        z=nngrillesimusin(betadensity,nbpt=nbpt,C=C,lzero=lzero,tunmax=tunmax)
    elif (prefixe=="sinus"):
        z=nngrillesimusin(sinusdensity,nbpt=nbpt,C=C,lzero=lzero,tunmax=tunmax)
        
    np.save(prefixe+"ntestsimusinN"+str(nbpt)+"C"+str(C)+"lzero"+str(lzero),z)
    return(z)

def nnpeinfmoy(intdens,la,La,mu,Mu,N=200,voir=False):
    def phi(t):
        return(La(t)-Mu(t)) #les fonctions en argument sont deja periodisees
    
    lesphi=np.array([phi(i/N) for i  in np.arange(2*N+1)])
    def peinf(i):
        t=i/N
        #print(i,t,la(t),mu(t),La(t),Mu(t))
        if (la(t) <= mu(t)):
            return(0.0)
        z=lesphi[i+1:i+N]- lesphi[i] #on regarde la courbe de taux sur une periode apres i
        #print("z=",z)
        if (z.min()<=0): #s 'il y a un piege, la proba d'extinctio nest nulle
            return(0.0) 
        else:
            return(1-(mu(t)/la(t)))
    ab=np.arange(N+1)
    val=np.array([peinf(i) for i in ab])
    intensite=np.array([intdens(i/N) for i in ab])
    def pf(t):
        "on fabrique la fonction a partir des valeurs entieres"
        return(val[int(N*t)])
    ppf=periodise(pf)
    pintens=periodise(intdens)
    if voir:
        abcisse=np.arange(0.0,2.0,1.0/(2*N))
        ordonnee=[ppf(t)*pintens(t) for t in abcisse]
        #print("ordonnee=",ordonnee)
        plt.plot(abcisse,ordonnee,label=r"$p_{e,\infty}$")
        plt.legend()
    #on fait une integration par la methode des trapezes c'e"st la que l'on utilise la densite d'introduction
    return((np.trapz(val*intensite,ab/N),ppf))

###################################################################
########### jeudi 14 février 2018 : lancons nous dans la figure

def intcontroltauxphietpe(lzero=2.1,tunmax=1.0,T=70,nbpt=100,C=0.2):
    r"on fixe a priori une grande periode"
    #couleurcontrole='lightblue'

    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
   #il faut ajuster le lzerosin pour que les processus aient le meme rzero

    def lasin(t):
        return(lzero*(1+np.sin(2*pi*t)))
    def Lasin(t):
        return(lzero*(t + (1/(2*pi))*(1-np.cos(2*pi*t))))
    def phisin(t):
        return(Lasin(t) -Mu(t))

    #print("phi(1),phisin(1)",phi(1),phisin(1))
    #on recupere les valeurs de tun et rhom qui donnent le minimum
    #pour la densite uniforme
    tun,rhom=minopt(nbpt,C,lzero,tunmax=1.0,prefixe="unif")
   
    #on definit les fonctions corespondant au controle optimal pour la sinusoide
    deltat=C/rhom
    tdeux=tun+deltat
    print("optimum sinus:tun,tdeux,rhom=",tun,tdeux,rhom)
    if (tdeux <=1):
        def rhosinopt(t):
            return(rhom if ((tun<t) and (t< tdeux)) else 0.0)
        def Larhosinopt(t):
            return(Lasin(t) - rhom*(Lasin(min(t,tdeux)) -Lasin(min(t,tun))))

    else:
        def rhosinopt(t):
            return(rhom if ((tun<t) or (t< tdeux-1)) else 0.0)
        def Larhosinopt(t):
            return(Lasin(t) - rhom*(Lasin(min(t,tdeux-1)) + Lasin(t) -Lasin(min(t,tun))))
    pmu=periodise(mu)
    plasin=periodise(lasin)
    
     #on recupere les valeurs de tun et rhom qui donnent le minimum
    #pour la densite non uniforme
    tuncerc,rhomcerc=minopt(nbpt,C,lzero,tunmax=1.0,prefixe="sinus")
   
    #on definit les fonctions corespondant au controle optimal pour la sinusoide
    deltatcerc=C/rhomcerc
    tdeuxcerc=tuncerc+deltatcerc
    print("optimum sinus densite sinus:tuncerc,tdeuxcerc,rhomcerc=",tuncerc,tdeuxcerc,rhomcerc)
    if (tdeuxcerc <=1):
        def rhosinoptcerc(t):
            return(rhom if ((tuncerc<t) and (t< tdeuxcerc)) else 0.0)
        def Larhosinoptcerc(t):
            return(Lasin(t) - rhomcerc*(Lasin(min(t,tdeuxcerc)) -Lasin(min(t,tuncerc))))

    else:
        def rhosinoptcerc(t):
            return(rhom if ((tuncerc<t) or (t< tdeuxcerc-1)) else 0.0)
        def Larhosinoptcerc(t):
            return(Lasin(t) - rhomcerc*(Lasin(min(t,tdeuxcerc-1)) + Lasin(t) -Lasin(min(t,tuncerc))))
    

    
    prhosinopt=periodise(rhosinopt)
    prhosinoptcerc=periodise(rhosinoptcerc)

    def plarhosinopt(t):
        return(plasin(t)*(1-prhosinopt(t)))
    pLarhosinopt=periodiseetintegre(Larhosinopt)      
    def plarhosinoptcerc(t):
        return(plasin(t)*(1-prhosinoptcerc(t)))
    pLarhosinoptcerc=periodiseetintegre(Larhosinoptcerc)      

 
    ############# la on va tracer les courbes    
    et=np.linspace(0,2,200)
    t=np.linspace(0,1,100)
    #plt.rcParams['figure.figsize'] = [16, 10]
    f,ax=plt.subplots(nrows=3, ncols=2, figsize=[16, 16])
    plt.tight_layout()
    #### d'abord les densités d'introduction

    #POUR L'UNIFORME
    plt.subplot(3,2,1)
    absticks()
    punifdensity=periodise(unifdensity)
    plt.ylabel(r"\bf{Introduction density}")
    plt.plot(et,[punifdensity(s) for s in et],color='black')

    #pour l'autre
    plt.subplot(3,2,2)
    absticks()
    psinusdensity=periodise(sinusdensity)
    plt.plot(et,[psinusdensity(s) for s in et],color='black')

    ####################################################
    plt.subplot(3,2,3)
    absticks()
    plt.yticks((0,1,2,3,4),(r"\bf{0}",r"\bf{1}",r"\bf{2}",r"\bf{3}",r"\bf{4}"))
    plt.ylabel(r"\bf{Birth and Death rates}")
    vplasin=np.array([plasin(s) for s in et])
    vplasinopt=np.array([plarhosinopt(s) for s in et])
    vplasinoptcerc=np.array([plarhosinoptcerc(s) for s in et])
    vmax=max(vplasin.max(),vplasinoptcerc.max(),vplasinopt.max())
    plt.plot(et,vplasin,label=r"$\lambda(t)$",color='black')
    plt.plot(et,[pmu(s) for s in et],label=r"$\mu(t)$",color='black',dashes=[2,2])
    plt.plot(et,vplasinopt,label=r"$\lambda_\rho(t)$",color=couleurcontrole,dashes=[1,1])
    plt.axis([0.0, 2.0,0.0, 1.1*vmax]) #par defaut dans matplotlib sans seaborn
    #plt.axis('scaled')
    #plt.legend()

    plt.subplot(3,2,4)
    absticks()
    plt.yticks((0,1,2,3,4),(r"\bf{0}",r"\bf{1}",r"\bf{2}",r"\bf{3}",r"\bf{4}"))
    plt.plot(et,vplasin,label=r"$\lambda(t)$",color='black')
    plt.plot(et,vplasinoptcerc,label=r"$\lambda_\rho(t)$",color=couleurcontrole,dashes=[1,1])
    plt.plot(et,[pmu(s) for s in et],label=r"$\mu(t)$",color='black',dashes=[2,2])
    plt.axis([0, 2.0, 0.0, 1.1*vmax]) 

    #plt.legend()




    #### proba emergence cas sinusoidal
    dcd=periodise(demicercledensity)
    plt.subplot(3,2,5)
    absticks()
    plt.yticks((0,0.2,0.4,0.6,0.8,1.0),(r"\bf{0}",r"\bf{0.2}",r"\bf{0.4}",r"\bf{0.6}",r"\bf{0.8}",r"\bf{1}"))
    plt.xlabel(r"\bf{Time of pathogen introduction} ${t_0}$")

    plt.ylabel(r"$p_e(t_0T,T)$")
    pemsin=nncalculepe(lasin,Lasin,mu,Mu,T,limit=100)
    ppemsin=periodise(pemsin)
    valppemsin=np.array([ppemsin(s) for s in et])
    plt.plot(et,valppemsin,label=r"$p_e$",color='black')
    vpemax=valppemsin.max()*((np.array([dcd(s) for s in et])).max())#on multiplie par le max de la densite du demi cercle
    plt.axis([0, 2.0, 0.0, 1.1*vpemax]) #par defaut dans matplotlib sans seaborn
    #mainenant pour le controle optimal
    pem=nncalculepe(plarhosinopt,Larhosinopt,mu,Mu,T,limit=100)
    ppem=periodise(pem)
    valppem=np.array([ppem(s) for s in et])
    plt.plot(et,valppem,label=r"$p_{e,opt}$",color=couleurcontrole,dashes=[1,1])
    a1,b1=trouvepremieretdernierzero(ppem,0.0,0.4)
    a2,b2=trouvepremieretdernierzero(ppem,0.5,1.0)
    bandegris(a1,b1,couleur=couleurcontrole,alpha=0.3)
    bandegris(a2,b2,couleur=couleurcontrole,alpha=0.3)
    bandegris(1+a2,1+b2,couleur=couleurcontrole,alpha=0.3)
    bandegris(1+a1,1+b1,couleur=couleurcontrole,alpha=0.3)
  
    #plt.legend()

    #### proba emergence cas sinusoidal avec autre densite
    plt.subplot(3,2,6)
    absticks()
    plt.yticks((0,0.2,0.4,0.6,0.8,1.0),(r"\bf{0}",r"\bf{0.2}",r"\bf{0.4}",r"\bf{0.6}",r"\bf{0.8}",r"\bf{1}"))
#plt.title(r"Emergence probability $p_e(t_0T,T)*density(t_0)$, T="+str(T))        
    plt.axis([0, 2.0, 0.0, 1.1*vpemax]) #par defaut dans matplotlib sans seaborn

    plt.xlabel(r"\bf{Time of pathogen introduction} ${t_0}$")
   
    plt.plot(et,[ppemsin(s)*dcd(s) for s in et],label=r"$p_e*density$",color='black') #la proba d'emergence sans controle
    pem=nncalculepe(plarhosinoptcerc,Larhosinoptcerc,mu,Mu,T)
    ppem=periodise(pem)
    valppem=np.array([ppem(s)*dcd(s) for s in et])
    plt.plot(et,valppem,label=r"$p_{e,opt}*density$",color=couleurcontrole,dashes=[1,1])
    a1,b1=trouvepremieretdernierzero(ppem,0.0,0.4)
    a2,b2=trouvepremieretdernierzero(ppem,0.5,1.0)
    bandegris(a1,b1,couleur=couleurcontrole,alpha=0.3)
    bandegris(a2,b2,couleur=couleurcontrole,alpha=0.3)
    bandegris(1+a2,1+b2,couleur=couleurcontrole,alpha=0.3)
    bandegris(1+a1,1+b1,couleur=couleurcontrole,alpha=0.3)

 
    #plt.legend()

    plt.savefig("intcontrolratesandemergenceproba.pdf",dpi=300)


    ##################################################
    ###############################################
    ######### Mercredi 27 févier 2019 : modélisation d'une vector borne disease

    #on modifie la fonction d'aproximation de la probabilite d'extinction en
    #utilisant la technique de Bacaer. Mais cette fois ci on passe en parametre
    #un vecteur des taux de mort et un tableau des taux de transition
#def vfbaca(Lambda,Mu,tau,T,nbpts=200,zzero=[1.0,1.0],tv=False):
def vfbaca(Lambda,Mu,T,tv=False,nbtau=10):
    tau=nbtau*T
    nbpts=10*tau
    d=len(Mu)
    zzero=np.array([1.0 for i in range(d)])
    
    def msisi(x,t):
        #s=tau-(t/T)
        s=(tau-t)/T
        return(np.array([-x[i]*Mu[i](s) + (1-x[i])*((np.array([Lambda[i][j](s)*x[j] for j in range(d)])).sum()) for i in range(d)]))

    timeint=np.arange(0,tau+1/nbpts,tau/nbpts)
    z=np.array(odeint(msisi,zzero,timeint))
    i=int(T*nbpts/tau)
    #print("T=",T,"tau=",tau,"z.shape",z.shape,"i=",i)
    res=z[-i:]
    if tv:
        plt.plot(timeint[-2*i:],z[-2*i:],label="z")
        plt.legend()
    return(res[::-1])

def modelsimple(ldu=1.0,ab=1.95,ep=1.0,mud=1.0,muu=1.0):
    def l21(x):
        return(ldu)
    def l12(x):
        return(ab*(1+ep*np.sin(2*np.pi*x)))
    def mu1(x):
        return(muu)
    def mu2(x):
        return(mud)
    def zero(x):
        return(0.0)

    Mu=[mu1,mu2]
    Lambda=[[zero,l12],[l21,zero]]
    return(Lambda,Mu)


def afvalue(a,x):
    return(np.array([a[i](x) for i in range(len(a))]))
def afvalue2(a,x):
    return(np.array([[a[i][j](x) for j in range(len(a[i]))] for i in range(len(a))]))
def ppos(x):
    return(0.5*(x+np.abs(x)))


def testvfbaca():
    Lambda,Mu=modelsimple()

    def peun(t):
        n=Lambda[0][1](t) *Lambda[1][0](t) -Mu[0](t)*Mu[1](t)
        d=(Lambda[0][1](t)+Mu[0](t)) *Lambda[1][0](t)
        return(ppos(n)/d)
    
    #nbpts=nb*tau
    z=vfbaca(Lambda,Mu,T=100,tv=False)
    #z=z.reshape(len(z),)
    t=np.linspace(0.0,1.0,len(z))
    vg=np.array([ppos(guess(afvalue2(Lambda,s),afvalue(Mu,s))) for s in t])
    plt.title("Simple 2D BD process")
    plt.plot(t,z,label=r"$p_e$")
    plt.plot(t,vg,label=r"guess")
    plt.plot(t,[peun(s) for s in t],label=r"$p_{e,1}$ direct")
    plt.legend()

def modelmoinssimple(ldu=1.0,ab=3.0,ep=1.0,mud=1.0,muu=1.0):
    def l21(x):
        return(ldu*(1+0.5*np.cos(2*np.pi*x)))
    def l12(x):
        return(ab*(1+ep*np.sin(2*np.pi*x)))
    def mu1(x):
        return(muu)
    def mu2(x):
        return(mud)
    def zero(x):
        return(0.0)

    Mu=[mu1,mu2]
    Lambda=[[zero,l12],[l21,zero]]
    return(Lambda,Mu)

def mtestvfbaca():
    Lambda,Mu=modelmoinssimple()

    def peun(t):
        n=Lambda[0][1](t) *Lambda[1][0](t) -Mu[0](t)*Mu[1](t)
        d=(Lambda[0][1](t)+Mu[0](t)) *Lambda[1][0](t)
        return(ppos(n)/d)
    
    #nbpts=nb*tau
    z=vfbaca(Lambda,Mu,T=100,tv=False)
    #z=z.reshape(len(z),)
    t=np.linspace(0.0,1.0,len(z))
    vg=np.array([ppos(guess(afvalue2(Lambda,s),afvalue(Mu,s))) for s in t])
    plt.title("Not so Simple 2D BD process")
    plt.plot(t,z[:,0],label=r"$p_e$")
    plt.plot(t,vg[:,0],label=r"guess")
    #plt.plot(t,[peun(s) for s in t],label=r"$p_{e,1}$ direct")
    plt.legend()



def tvfbacadimun(T=50,lzero=2.1,loffset=0.0):
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
   #il faut ajuster le lzerosin pour que les processus aient le meme rzero

    def lasin(t):
        return(loffset+lzero*(1+np.sin(2*pi*t)))
    def Lasin(t):
        return(loffset*t+lzero*(t + (1/(2*pi))*(1-np.cos(2*pi*t))))
    def guess(t):
        a=lasin(t)
        if (a>1):
            return(1-(1/a))
        else:
            return(0.0)

    #nbpts=nb*tau
    Lambda=[[lasin]]
    VMu=[mu]
    z=vfbaca(Lambda,VMu,T=T,tv=False)
    z=z.reshape(len(z),)
    pem=nncalculepe(lasin,Lasin,mu,Mu,T)
    t=np.linspace(0.0,1.0,len(z))
    valpem=np.array([pem(s) for s in t])
    vg=np.array([guess(s) for s in t])
    plt.title("Linear BD,sinusoidal birth rate")
    plt.plot(t,z,label="simulation de Bacaer")
    plt.plot(t,valpem,label="calcul integral exact")
    plt.plot(t,vg,label=r"guess : $(1-\lambda/\mu)^+$")
    plt.legend()
    print("difference max",np.abs(valpem-z).max())
    #on n'a pas zero car les tableaux n'ont pas meme dimension
    return((z,valpem))

def produit(liste):
    if (len(liste)==0):
        return(1)
    else:
        return(reduce((lambda x,y : x*y),liste))

def guess(lam,mu):
    r"lam et mu sont un tableau et un vecteur de taux : retourne  les probabilites d extinction"
    d=len(mu)
    numerateur=produit([lam[i%d,(i+1)%d] for i in range(d)]) -produit(mu)
    if (numerateur<=0.0):
        return(np.array([0.0 for i in range(d)]))
    denominateur=np.array([np.array([produit([mu[j%d] for j in range(i,i+k)])*produit([lam[i%d,(i+1)%d] for i in range(i+k,i+d)]) for k in range(d)]).sum() for i in range(d)])
    #print("lam,mu,numerateur,denominateur",lam,mu,numerateur,denominateur)
    return(numerateur/denominateur)

def rzerovar(lam,mu):
    d=len(mu)
    numerateur=produit([lam[i%d,(i+1)%d] for i in range(d)])
    return(numerateur/produit(mu))
            
def testeguess():
    lam=np.array([[0,1],[2,3]])
    mu=np.array([1,1])
    return(guess(lam,mu))

def lambrecht(t):
        return(0.001044*t*(t-12.286)*np.sqrt(ppos(32.461-t)))

def tlbrech():
    t=np.linspace(15,35,100)
    plt.plot(t,[lambrecht(s) for s in t])
    plt.title("Transmission probability as a function of temperature :\n the model of  Lambrecht et al.")
    plt.xlabel("Temperature")
    plt.savefig("Lambrecht.pdf",dpi=300,bbox_inches='tight',pad_inches=0.2)


def fsamodel(Tmoy=25,DT=20):
    muunc=(1/6.0)+1.0/(75*365)
    av=0.25
    phihv=0.5
    nvsnh=5 #le rapport Nv/Nh
    def l34(x):
        return(1/9.0)
    def l12(x):
        return(1/6.0)
    def l23(x):
        return(av*phihv*nvsnh)
    def mu1(x):
        return(muunc)
    def mu2(x):
        return(muunc)
    def mu3(x):
        return((1.0/5) + (1.0/9.0))
    def mu4(x):
        return(1.0/9)
    def zero(x):
        return(0.0)
    def temperature(t):
        return(Tmoy+0.5*DT*np.sin(2*pi*t))
    def l41(t):
        return(av*lambrecht(temperature(t)))
    
    Mu=[mu1,mu2,mu3,mu4]
    Lambda=[[zero,l12,zero,zero],[zero,zero,l23,zero],[zero,zero,zero,l34],[l41,zero,zero,zero]]
    return(Lambda,Mu)


def tfsa(DT=10,T=365,nbtau=5):
    Lambda,Mu=fsamodel(DT=DT)
    z=vfbaca(Lambda,Mu,T=T,tv=False,nbtau=nbtau)
    t=np.linspace(0.0,1.0,len(z))
    vg=np.array([ppos(guess(afvalue2(Lambda,s),afvalue(Mu,s))) for s in t])
    plt.title("FSA process : emergence probability for one exposed human")
    plt.plot(t,z[:,0],label=r"$p_e$")
    #plt.plot(t,z,label=r"$p_e$")
    #plt.plot(t,[Lambda[3][0](s) for s in t],label=r"$\lambda_{I_V,E_H}$")
    plt.plot(t,vg[:,0],label=r"guess")
    #plt.plot(t,vg,label=r"guess")
    plt.xlabel("Introduction time")
    plt.legend()
    return(z)


############ Vendredi 1er Mars 2019 : modele SA amerique du sud de Zhang et al

def tmosden(Tmoy=25,Topt=28,DT=10,damping=50):
    
    def temperature(t):
        return(Tmoy+0.5*DT*np.sin(2*pi*t))
    def denfunc(t):
        return(np.exp(-(t-Topt)**2/damping))
    def mosquitodensity(t):
        return(denfunc(temperature(t)))

    f,ax=plt.subplots(nrows=3, ncols=1, figsize=[8, 12])
    t=np.linspace(0,1,100)

    plt.subplot(3,1,1)
    plt.plot(t,[temperature(s) for s in t],label="temperature")
    plt.title("Temperature")
    plt.xlabel("relative time in the year")
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(t,[10*mosquitodensity(s) for s in t],label="mosquitodensity")
    plt.title("\n Mosquito density vs temperature : Zhang et al model")
    plt.xlabel("relative time in the year")
    plt.legend()

    plt.subplot(3,1,3)
    tempabs=np.linspace(Tmoy-0.5*DT,Tmoy+0.5*DT,100)
    vdf=np.array([denfunc(s) for s in tempabs])
    print("denfunc, max, min,ratio",vdf.max(),vdf.min(),vdf.max()/vdf.min())
    plt.plot(tempabs,[denfunc(s) for s in tempabs],label="mosquitodensity")
    plt.xlabel(r"Temperature $T$")
    plt.legend()

def samodel(Tmoy=25,Topt=25,DT=20,muv=0.4,kc=10):
    betatilde=0.3
    epsilonv=1.0/8
    epsilonh=1.0/7
    muh=0.2
    #muv=1.0
    damping=5.3*5.3
    def l34(x):
        return(epsilonv)
    def l12(x):
        return(epsilonh)
    def l23(x):
        return(betatilde*kc*np.exp(-(temperature(x)-Topt)**2/damping))
    def mu1(x):
        return(epsilonh)
    def mu2(x):
        return(muh)
    def mu3(x):
        return(epsilonv+muv)
    def mu4(x):
        return(muv)
    def zero(x):
        return(0.0)
    def temperature(t):
        return(Tmoy+0.5*DT*np.sin(2*pi*t))
    def l41(t):
        return(betatilde)
    
    Mu=[mu1,mu2,mu3,mu4]
    Lambda=[[zero,l12,zero,zero],[zero,zero,l23,zero],[zero,zero,zero,l34],[l41,zero,zero,zero]]
    return(Lambda,Mu)


def tsa(Tmoy=25,Topt=28,DT=20,T=365,nbtau=5,muv=0.5,kc=10):
    def temperature(t):
        return(Tmoy+0.5*DT*np.sin(2*pi*t))
    Lambda,Mu=samodel(Tmoy=Tmoy,Topt=Topt,DT=DT,muv=muv,kc=kc)
    z=vfbaca(Lambda,Mu,T=T,tv=False,nbtau=nbtau)

    #f,ax1=plt.subplots(figsize=[8, 12])
    f,ax1=plt.subplots()

    t=np.linspace(0.0,1.0,len(z))
    vg=np.array([ppos(guess(afvalue2(Lambda,s),afvalue(Mu,s))) for s in t])
    lud=Lambda[1][2]
    lihev=np.array([lud(s) for s in t])
    
    ax1.set_title("Emergence probability for one exposed human")
    ax1.plot(t,z[:,0],label=r"$p_e$")
    #plt.plot(t,[temperature(s) for s in t],label=r"temperature")
    ax1.plot(t,vg[:,0],label=r"guess",color='b')
    ax1.set_xlabel("Introduction time")
    ax1.set_ylabel(r"$p_e$",color='b')
    ax1.tick_params('y',color='b')
    ax1.legend()

    x=np.array([rzerovar(afvalue2(Lambda,s),afvalue(Mu,s)) for s in t])
    
    #k=kde.gaussian_kde(x)
    # plt.plot(x,k(x),label=r"density estimation")
    # plt.xlabel(r"$R_0$")
    # plt.legend()

    ax2=ax1.twinx()
    ax2.set_ylabel(r"time varying $R_0$",color='r')
    ax2.plot(t,lihev,label=r"$\lambda_{1,2}$",color='green')

    ax2.plot(t,x,color='r',label=r"$R_0$")
    ax1.tick_params('y',color='r')
    ax2.legend()
    plt.savefig("SA1.pdf",dpi=300,bbox_inches='tight',pad_inches=0.2)
    
    plt.show()
    return(z)


################ mrecredi 6 mars 2019 : emergence for a vector borne disease
########### essayons de mettre en place une strategie de controle
############ pour le cas ou la periodicite est uniquement dans le rapport nh/nv
########### on diminue ce facteur de rhom pendant l'intervalle de temps t1,t2

def tsacontrol(Tmoy=25,DT=8,T=365,nbtau=5,muv=0.5,tun=0.4,rhom=0.9,C=0.2,kc=10):
    def temperature(t):
        return(Tmoy+0.5*DT*np.sin(2*pi*t))
    Lambda,Mu=samodel(Tmoy=Tmoy,DT=DT,muv=muv,kc=kc)
    z=vfbaca(Lambda,Mu,T=T,tv=False,nbtau=nbtau)

    fC=modelsacontrolled(Tmoy=Tmoy,DT=DT,muv=muv,kc=kc,C=C)
    LambdaC,MuC=fC(tun,rhom)
    zC=vfbaca(LambdaC,MuC,T=T,tv=False,nbtau=nbtau)

    #f,ax1=plt.subplots(figsize=[8, 12])
    f,ax=plt.subplots(nrows=2,ncols=2, figsize=[11, 16])

    t=np.linspace(0.0,1.0,len(z))
    vg=np.array([ppos(guess(afvalue2(Lambda,s),afvalue(Mu,s))) for s in t])
    vgC=np.array([ppos(guess(afvalue2(LambdaC,s),afvalue(MuC,s))) for s in t])
    lud=Lambda[1][2]
    #print("lud",lud)
    lihev=np.array([lud(s) for s in t])
    ludC=LambdaC[1][2]
    lihevC=np.array([ludC(s) for s in t])
    
    plt.subplot(2,2,1)
    plt.title(r"Transmission rate $\lambda_{I^H,E^V}$")
    plt.plot(t,lihev,color='black')

    
    plt.subplot(2,2,3)
    
    plt.title("Emergence probability for one exposed human")
    plt.plot(t,z[:,0],label=r"$p_e$",color='black')
    #plt.plot(t,[temperature(s) for s in t],label=r"temperature")
    plt.plot(t,vg[:,0],label=r"guess",color='black',dashes=[6,2])
    plt.xlabel("Introduction time")
    plt.legend()

    plt.subplot(2,2,2)
    plt.title(r"Controlled Transmission rate $\lambda_{I^H,E^V}$")
    plt.plot(t,lihevC,color='black')


    plt.subplot(2,2,4)
    
    plt.title("Controlled Emergence probability for one exposed human")
    plt.plot(t,zC[:,0],label=r"$p_e$",color='black')
    #plt.plot(t,[temperature(s) for s in t],label=r"temperature")
    plt.plot(t,vgC[:,0],label=r"guess",color='black',dashes=[6,2])
    plt.xlabel("Introduction time")
    plt.legend()

    
    plt.savefig("SA1control.pdf",dpi=300,bbox_inches='tight',pad_inches=0.2)




    

def modelsacontrolled(Tmoy=25,DT=20,muv=0.4,kc=10,C=0.2):
    r""" retourne une fonction qui prend pour parametre tun et rhom et retourne les parametres controles, c'est a dire les fonctions lambbda et mu avec la diminution du facteur kc entre tun et tdeux"""
    betatilde=0.3
    epsilonv=1.0/8
    epsilonh=1.0/7
    muh=0.2
    #muv=1.0
    damping=5.3*5.3
    def l34(x):
        return(epsilonv)
    def l12(x):
        return(epsilonh)
    def mu1(x):
        return(epsilonh)
    def mu2(x):
        return(muh)
    def mu3(x):
        return(epsilonv+muv)
    def mu4(x):
        return(muv)
    def zero(x):
        return(0.0)
    def temperature(t):
        return(Tmoy+0.5*DT*np.sin(2*pi*t))
    def l41(t):
        return(betatilde)
     
    def sinqextmoy(tun,rhom):
        #on peut prendre tdeux >=1, mais il faut imposer rhom >= C
        #et prendre les fonctions periodisees
        assert(rhom >= C)
        deltat=C/rhom
        tdeux=tun+deltat
        #print("tun,tdeux,rhom",tun,tdeux,rhom)
        if (tdeux <=1):
            def facteur(x):
                if ((x<=tun) or (x>=tdeux)):
                    return(1.0)
                else:
                    return(1-rhom)
        else:
            def facteur(x):
                if ((x<= tdeux -1) or (x>=tun)):
                    return(1-rhom)
                else:
                    return(1.0)
        #print("facteur x=milieu tun,tdeux",facteur(0.5*(tun+tdeux)))
        def l23(x):
                return(facteur(x)*betatilde*kc*np.exp(-(temperature(x)-Tmoy)**2/damping))
        
   
        Mu=[mu1,mu2,mu3,mu4]
        Lambda=[[zero,l12,zero,zero],[zero,zero,l23,zero],[zero,zero,zero,l34],[l41,zero,zero,zero]]
        return(Lambda,Mu)

    return(sinqextmoy)



def samodelbis(Tmoy=25,DT=20,muv=0.4,kc=10,betatilde=0.3):
    r""" on remplace le betatilde, proba de transmission fixe, par une fonction de Lambrecht"""
    #betatilde=0.3
    epsilonv=1.0/8
    epsilonh=1.0/7
    muh=0.2
    #muv=1.0
    damping=5.3*5.3
    def l34(x):
        return(epsilonv)
    def l12(x):
        return(epsilonh)
    def l23(x):
        return(2*betatilde*lambrecht(temperature(x))*kc*np.exp(-(temperature(x)-Tmoy)**2/damping))
    def mu1(x):
        return(epsilonh)
    def mu2(x):
        return(muh)
    def mu3(x):
        return(epsilonv+muv)
    def mu4(x):
        return(muv)
    def zero(x):
        return(0.0)
    def temperature(t):
        return(Tmoy+0.5*DT*np.sin(2*pi*t))
    def l41(t):
        #modif du 22 mai
        #return(2*betatilde*lambrecht(temperature(t)))
        return(lambrecht(temperature(t)))
    
    Mu=[mu1,mu2,mu3,mu4]
    Lambda=[[zero,l12,zero,zero],[zero,zero,l23,zero],[zero,zero,zero,l34],[l41,zero,zero,zero]]
    return(Lambda,Mu)



def modelsacontrolledbis(Tmoy=25,DT=20,muv=0.4,kc=10,C=0.2,betatilde=0.3):
    r""" retourne une fonction qui prend pour parametre tun et rhom et retourne les parametres controles, c'est a dire les fonctions lambbda et mu avec la diminution du facteur kc entre tun et tdeux : pour le modele bis"""
    #betatilde=0.3
    epsilonv=1.0/8
    epsilonh=1.0/7
    muh=0.2
    #muv=1.0
    damping=5.3*5.3
    def l34(x):
        return(epsilonv)
    def l12(x):
        return(epsilonh)
    def mu1(x):
        return(epsilonh)
    def mu2(x):
        return(muh)
    def mu3(x):
        return(epsilonv+muv)
    def mu4(x):
        return(muv)
    def zero(x):
        return(0.0)
    def temperature(t):
        return(Tmoy+0.5*DT*np.sin(2*pi*t))
    def l41(t):
        return(2*betatilde*lambrecht(temperature(t)))
     
    def sinqextmoy(tun,rhom):
        #on peut prendre tdeux >=1, mais il faut imposer rhom >= C
        #et prendre les fonctions periodisees
        assert(rhom >= C)
        deltat=C/rhom
        tdeux=tun+deltat
        #print("tun,tdeux,rhom",tun,tdeux,rhom)
        if (tdeux <=1):
            def facteur(x):
                if ((x<=tun) or (x>=tdeux)):
                    return(1.0)
                else:
                    return(1-rhom)
        else:
            def facteur(x):
                if ((x<= tdeux -1) or (x>=tun)):
                    return(1-rhom)
                else:
                    return(1.0)
        #print("facteur x=milieu tun,tdeux",facteur(0.5*(tun+tdeux)))
        def l23(x):
                return(facteur(x)*2*betatilde*lambrecht(temperature(x))*kc*np.exp(-(temperature(x)-Tmoy)**2/damping))
        
   
        Mu=[mu1,mu2,mu3,mu4]
        Lambda=[[zero,l12,zero,zero],[zero,zero,l23,zero],[zero,zero,zero,l34],[l41,zero,zero,zero]]
        return(Lambda,Mu)

    return(sinqextmoy)


def tsacontrolbis(Tmoy=25,DT=8,T=365,nbtau=5,muv=0.5,tun=0.4,rhom=0.9,C=0.2,kc=10,betatilde=0.3):
    def temperature(t):
        return(Tmoy+0.5*DT*np.sin(2*pi*t))
    Lambda,Mu=samodelbis(Tmoy=Tmoy,DT=DT,muv=muv,kc=kc,betatilde=betatilde)
    z=vfbaca(Lambda,Mu,T=T,tv=False,nbtau=nbtau)

    fC=modelsacontrolledbis(Tmoy=Tmoy,DT=DT,muv=muv,kc=kc,C=C,betatilde=betatilde)
    LambdaC,MuC=fC(tun,rhom)
    zC=vfbaca(LambdaC,MuC,T=T,tv=False,nbtau=nbtau)


    t=np.linspace(0.0,1.0,len(z))
    def guessori(s):
        return(ppos(guess(afvalue2(Lambda,s),afvalue(Mu,s))))
    def guesscontrol(s):
        return(ppos(guess(afvalue2(LambdaC,s),afvalue(MuC,s))))
    
    vg=np.array([guessori(s) for s in t])
    vgC=np.array([guesscontrol(s) for s in t])
    lud=Lambda[1][2]
    #print("lud",lud)
    lihev=np.array([lud(s) for s in t])
    ludC=LambdaC[1][2]
    lihevC=np.array([ludC(s) for s in t])


    
    lquatreun=Lambda[3][0]
    liveh=np.array([lquatreun(s) for s in t])
    lquatreunC=LambdaC[3][0]
    livehC=np.array([lquatreunC(s) for s in t])
    
    #f,ax1=plt.subplots(figsize=[8, 12])
    f,(ax1,ax2,ax3)=plt.subplots(nrows=3,ncols=1, figsize=[16,20])
    f.subplots_adjust(top=0.9,bottom=0.1,left=0.125,right=0.9,hspace=0.5)
    
    mettrelettreax(ax1,'A')
    mettrelettreax(ax2,'B')
    mettrelettreax(ax3,'C')

    #plt.subplot(2,1,2)
    for a in (ax1,ax2,ax3):
        axdemiabsticks(a)

    ax3.axis([0.0,1.0,0.0,0.4])
    ax2.axis([0.0,1.0,0.0,4.0])
    ax1.axis([0.0,1.0,0.6,1.0])
    
    ax3.set_ylabel(r"\bf{Emergence probability}",fontsize=22)
    ax3.plot(t,z[:,0],label=r"$p_e$",color='black')
    #plt.plot(t,vg[:,0],label=r"guess",color='black',dashes=[6,2]) #le 4 juin
    ax3.set_xlabel(r"\bf{Time of Pathogen introduction} ${t_0}$",fontsize=22)

    ax3.plot(t,zC[:,0],label=r"$p_e$",color=couleurcontrole,linestyle=':')
    #plt.plot(t,vgC[:,0],label=r"guess",dashes=[6,2],color=couleurcontrole) 

    #a1,b1=trouvepremieretdernierzero(guesscontrol,0.0,1.0,tolerance=0.01)
    def pemcontrol(s):
        N=len(zC[:,0])-1
        i=int(s*N)
        return(zC[i,0])
    a1,b1=trouvepremieretdernierzero(pemcontrol,0.0,1.0,tolerance=0.03)
    tdeux=tun+(C/rhom)
    axbandegris(ax3,a1,tun,couleur=couleurcontrole,alpha=0.15)
    axbandegris(ax3,tun,tdeux,couleur=couleurcontrole,alpha=0.3)
    
    #plt.subplot(2,1,1)
    ax2.set_ylabel(r"$\lambda_{I^H,E^V}$",fontsize=30)
    #ax2.set_ylabel(r"\bf{Transmission rate} $\lambda_{I^H,E^V}$",fontsize=22)
    ax2.plot(t,lihev,color='black')
    ax2.plot(t,lihevC,color=couleurcontrole,linestyle=':')

    #ax1.set_ylabel(r"\bf{Transmission rate} $\lambda_{I^V,E^H}$",fontsize=22)
    ax1.set_ylabel(r" $\lambda_{I^V,E^H}$",fontsize=30)
    ax1.plot(t,liveh,color='black')
   
    
    plt.savefig("SA2control.pdf",dpi=300,bbox_inches='tight',pad_inches=0.2)



####################################"" Jeudi 14 mars
################## trace de la heatmap de pe dans le cas step
#################" pour voir qu'il ya a toute une variete de strategies optimales



####  vendredi premier février 2019 : attaquons nous a la seconde figure

#tout d'abord on reprend notre calcul de la strategie de controle optimale
    
def nsimustep(lzero=4.0,C=0.2,voir=False,gamma=0.3):
    def la(t):
        if (t< 1-gamma):
            return(lzero)
        else:
            return(0.0)
    def La(t):
        if (t< 1-gamma):
            return(lzero*t)
        else:
            return(lzero*(1-gamma))
 
        return(lzero*(1+np.sin(2*pi*t)))
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    pla=periodise(la)
    pmu=periodise(mu)
    pLa=periodiseetintegre(La)
    pMu=periodiseetintegre(Mu)
    
    def sinqextmoy(tun,rhom,C=C):
        #on peut prendre tdeux >=1, mais il faut imposer rhom >= C
        #et prendre les fonctions periodisees
        assert(rhom >= C)
        deltat=C/rhom
        tdeux=tun+deltat
        if (tdeux <=1):
            def rho(t):
                return(rhom if ((tun<t) and (t< tdeux)) else 0.0)
            def Larho(t):
                return(La(t) - rhom*(La(min(t,tdeux)) -La(min(t,tun))))

        else:
            def rho(t):
                return(rhom if ((tun<t) or (t< tdeux-1)) else 0.0)
            def Larho(t):
                return(La(t) - rhom*(La(min(t,tdeux-1)) + La(t) -La(min(t,tun))))
           
        prho=periodise(rho)
            
        def plarho(t):
            return(pla(t)*(1-prho(t)))
        pLarho=periodiseetintegre(Larho)      

        v=npeinfmoy(plarho,pLarho,pmu,pMu,voir=voir)
        return(v)
    return(sinqextmoy)
                   
def ngrillesimustep(nbpt,lzero=4.0,C=0.2,gamma=0.3,tunmax=1.0):
    #print("ngrillesimusin,lzero=",lzero,",C=",C)
    f=nsimustep(C=C,lzero=lzero,gamma=gamma)
    tun=np.arange(0,tunmax,tunmax/nbpt)
    rho=(1.0/nbpt)+np.arange(C,1.0,1.0/nbpt)
    zs = np.array([[f(i,j) for j in rho] for i in tun])
    return(zs)

def ntestsimustep(nbpt=100,C=0.3,lzero=4.0,gamma=0.3,tunmax=1.0):
    lzeromodif=round(lzero*100)/100
    z=ngrillesimustep(nbpt=nbpt,C=C,lzero=lzero,tunmax=tunmax,gamma=gamma)
    np.save("ntestsimustepN"+str(nbpt)+"C"+str(C)+"lzero"+str(lzero),z)
    return(z)

def mettrelettreax(ax,lettre,gauche=-0.1,taille=60):
    mot=r"\textbf{%s}" %(lettre,)
    ax.text(gauche, 1.1, mot, transform=ax.transAxes, 
            fontsize=taille)    
def enlevecadreax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    



############# le 26 juin : pour des raisons de test
def nnstepvoir(z,zsin,C=0.2,tunmax=1.0,beamer=False,vrhomtun=False,gamma=0.3,vopt=True,tunstepopt=0.2,rhomstepopt=0.85,rzeropt=1.28,tunrzeropt=0.15,rhomrzeropt=1.0):
    N=(z.shape)[0]
    tun=np.arange(0.0,tunmax,tunmax/N)
    rho=(1.0/N)+np.arange(C,1.0,1.0/N)
    X,Y=np.meshgrid(rho,tun)
    if (beamer):
        #fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=[18, 8])
        fig=plt.figure(figsize=[22, 8])
        gs = gridspec.GridSpec(1, 2,width_ratios=[1,1.28])

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        #fig, (ax1,ax2) = plt.subplots(1,2,figsize=[18, 8])
    else:
        fig, (ax1,ax2,ax3) = plt.subplots(1,2,figsize=[12, 16])
    #plt.tight_layout()
    #plt.subplots_adjust(left=0.2)
    
    #plt.subplot(1,2,1)
    ax1.set_title(r"\bf{Square Wave}",fontsize=40)
    ax2.set_title(r"\bf{Sinusoidal Wave}",fontsize=40)
    mettrelettreax(ax1,'A')
    enlevecadreax(ax1)
    ax1.axis([C,1.1,0,1.0])
    zmin=z.min()
    zmax=z.max()
    lec=np.array(trouvecontour(z,tun,rho))
    #ax1.plot(lec[:,0],lec[:,1],color='red',linewidth=5,linestyle=':')
    ax1.plot(lec[:,0],lec[:,1],color='red',linestyle=':')
    ax1.hlines(y=0.001,color='red',linestyle=':',xmin=0.67,xmax=1.0)
    ax1.vlines(x=0.999,color='red',linestyle=':',ymin=0.0,ymax=0.5)

    if (vrhomtun):
        rhomrestreint=np.arange(C/(1-gamma),1,1/N)
        ax1.plot(rhomrestreint,[(1-gamma - (C/s)) for s in rhomrestreint],color='lightgreen',linewidth=10)

    #c = ax1.pcolormesh(X,Y,z,cmap=Lacolormap,vmin=zmin,vmax=zmax)
    #c = ax1.pcolormesh(X,Y,z,cmap='RdBu',vmin=zmin,vmax=zmax)
    c = ax1.pcolormesh(X,Y,z,cmap='viridis',vmin=zmin,vmax=zmax)
    #fig.colorbar(c,ax=ax1)
    ax1.set_xlabel(r"\par\bf{Intensity of control} ${\rho_M}$",fontsize=40,labelpad=20)
    ax1.set_ylabel(r"\bf{Start of control} $t_1$",fontsize=40)
    ax1.set_yticks((0.0,0.2,0.4,0.6,0.8))
    ax1.set_yticklabels((r"\bf{0}",r"\bf{0.2}",r"\bf{0.4}",r"\bf{0.6}",r"\bf{0.8}"))
    ax1.set_xticks((0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0))
    ax1.set_xticklabels((r"\bf{0.2}",r"\bf{0.3}",r"\bf{0.4}",r"\bf{0.5}",r"\bf{0.6}",r"\bf{0.7}",r"\bf{0.8}",r"\bf{0.9}",r"\bf{1.0}"))
    ax1.axis([C,1.1,0,1.0])

    m=z.min()
    y=np.where(z==m)
    xopt=tun[y[0][0]]
    yopt=rho[y[1][0]]
    titre=r"Step case, Mean emergence probability: Cost="+str(C)
    if vopt:
        ax1.plot(rhomstepopt,tunstepopt,marker='x',color=couleurcontrole,markersize=30,mew=4)
    #maintenant on s'occupe de la heatmap pour la sinusoide
    #plt.subplot(1,2,2)
    N=(zsin.shape)[0]
    
    X,Y=np.meshgrid(rho,tun)
    mettrelettreax(ax2,'B')
    enlevecadreax(ax2)
        
    zsinmin=zsin.min()
    zsinmax=zsin.max()
    #c = ax2.pcolormesh(X,Y,zsin,cmap=Lacolormap,vmin=zsinmin,vmax=zsinmax)
    c = ax2.pcolormesh(X,Y,zsin,cmap='viridis',vmin=zmin,vmax=zmax)
    fig.colorbar(c,ax=ax2,pad=0.05)
    ax2.text(0.97,1.05,r"$<p_e>$", transform=ax2.transAxes, 
             fontsize=40) 
    #ax3.set_title(r"$<p_e>$",fontsize=30)
    #ax3.axis([0.0,0.0,0.0,0.0])
    #enlevecadreax(ax3)

    #ax2.set_ylabel(r"\bf{Start of control} ${t_1}$",fontsize=40)
    ax2.set_xlabel(r"\bf{Intensity of control} ${\rho_M}$",fontsize=40,labelpad=10)
    ax2.set_yticks((0.0,0.2,0.4,0.6,0.8))
    ax2.set_yticklabels((r"\bf{0}",r"\bf{0.2}",r"\bf{0.4}",r"\bf{0.6}",r"\bf{0.8}"))
    ax2.set_xticks((0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0))
    ax2.set_xticklabels((r"\bf{0.2}",r"\bf{0.3}",r"\bf{0.4}",r"\bf{0.5}",r"\bf{0.6}",r"\bf{0.7}",r"\bf{0.8}",r"\bf{0.9}",r"\bf{1.0}"))
    ax2.axis([C,1.1,0,1.0])
    m=zsin.min()
    y=np.where(zsin==m)
    xopt=tun[y[0][0]]
    yopt=rho[y[1][0]]
    ax2.plot(rhomrzeropt,tunrzeropt,marker='x',color='red',markersize=30,mew=4)
    ax2.plot(yopt,xopt,marker='x',color=couleurcontrole,markersize=30,mew=4)

    
    plt.savefig("heatmapstepandsinus.pdf",dpi=300,bbox_inches='tight',pad_inches=0.2)
    plt.show()


    ################# Jeudi 14 mars : maintenant on montre l'influence de la temperature locale et surtout quand elle est diferenete de la temperature otimale de developpement des vecteurs

def tsageo(Tmoy1=25,Tmoy2=28,Topt=28,DT=8,T=365,nbtau=5,muv=0.4,tun=0.4,rhom=0.9,C=0.2,kc=10,complet=False,beamer=True):
    def flechewinter(hauteur,xmin,xmax,couleur=cgris):
        hdelta=0.05
        #plt.arrow(xmax-hdelta,hauteur,-(xmax-xmin)+2*hdelta,0,shape='full',  length_includes_head=True,color=couleur,alpha=1.0,linewidth=15,head_starts_at_zero=True,head_width=5e-2)
        plt.arrow(xmax,hauteur,-(xmax-xmin)+2*hdelta,0,shape='full',  length_includes_head=True,color=couleur,alpha=1.0,linewidth=40,head_starts_at_zero=True,head_width=2e-2)
    def temperature(t):
        return(Tmoy+0.5*DT*np.sin(2*pi*t))
    Lambda1,Mu1=samodel(Tmoy=Tmoy1,Topt=Topt,DT=DT,muv=muv,kc=kc)
    z1=vfbaca(Lambda1,Mu1,T=T,tv=False,nbtau=nbtau)

    Lambda2,Mu2=samodel(Tmoy=Tmoy2,Topt=Topt,DT=DT,muv=muv,kc=kc)
    z2=vfbaca(Lambda2,Mu2,T=T,tv=False,nbtau=nbtau)

    #f,ax1=plt.subplots(figsize=[8, 12])
    nblignes= 2
    if (beamer):
        f,ax=plt.subplots(nrows=nblignes,ncols=2, figsize=[16, 10])
    else:
        f,ax=plt.subplots(nrows=nblignes,ncols=2, figsize=[10, 16])
        
    #plt.tight_layout()
    plt.subplots_adjust(left=0.065,right=0.97,bottom=0.1,top=0.9,hspace=0.5)
    plt.rc('text', usetex=True)
    

    t=np.linspace(0.0,1.0,len(z1))
    vg1=np.array([ppos(guess(afvalue2(Lambda1,s),afvalue(Mu1,s))) for s in t])
    vg2=np.array([ppos(guess(afvalue2(Lambda2,s),afvalue(Mu2,s))) for s in t])
    lud1=Lambda1[1][2]
    #print("lud",lud)
    lihev1=np.array([lud1(s) for s in t])
    lud2=Lambda2[1][2]
    lihev2=np.array([lud2(s) for s in t])
    vmin=min(lihev1.min(),lihev2.min())
    vmax=max(lihev1.max(),lihev2.max())
    def fguess2(x):
        if (x<1):
            #print("int(x*len(z1))",int(x*len(z1)))
            return(vg2[int(x*len(z1)),0])
        else:
            return(vg2[-1,0])
    awinter,bwinter=trouvepremieretdernierzero(fguess2,0.5,1.0)
     
    #plt.subplot(nblignes,2,1)
    axc=ax[0,0]
    axdemiabsticks(axc)
    mettrelettreax(axc,'A',taille=50,gauche=0)
    axc.axis([0, 1.0, 0.0,4.0])
    axc.set_yticks((0.0, 1.0,2.0,3.0,4.0))
    axc.set_yticklabels((r'\bf{0}', r'\bf{1}', r'\bf{2}',r'\bf{3}',r'\bf{4}'), color='k', size=20)
    axc.set_ylabel( r"\bf{Transmission rate} $\lambda_{I^V,E^H}$",fontsize=22)
    axc.plot(t,lihev1,color='black')

    #plt.subplot(nblignes,2,2)
    axc=ax[0,1]
    axdemiabsticks(axc)
    mettrelettreax(axc,'B',taille=50)
    axc.axis([0, 1.0, 0.0,4.0])
    axc.set_yticks((0.0, 1.0,2.0,3.0,4.0))
    axc.set_yticklabels((r'\bf{0}', r'\bf{1}', r'\bf{2}',r'\bf{3}',r'\bf{4}'), color='k', size=20)
    axc.plot(t,lihev2,color='black')
    axbandegris(axc,awinter,bwinter)
    
    #plt.subplot(nblignes,2,3)
    axc=ax[1,0]
    axdemiabsticks(axc)
    mettrelettreax(axc,'C',taille=50,gauche=0)
    axc.axis([0, 1.0, 0.0,0.4])
    axc.set_yticks((0.0, 0.1,0.2,0.3))
    axc.set_yticklabels((r'\bf{0}', r'\bf{0.1}', r'\bf{0.2}',r'\bf{0.3}'), color='k', size=20)
    if (not(complet)):
        axc.set_ylabel(r"\bf{Emergence probability}",fontsize=22)
        axc.plot(t,z1[:,0],label=r"$p_e$",color='black')
    else:
        axc.set_ylabel(r"\bf{Emergence probabilities} ${p_{e,i}}$",fontsize=22)
        axc.plot(t,z1[:,0],label=r"Exposed Human",color='black')
        axc.plot(t,z1[:,1],label=r"Infectious Human",color='black',dashes=[2,2])
        axc.plot(t,z1[:,2],label=r"Exposed Vector",color='red')
        axc.plot(t,z1[:,3],label=r"Infectious Vector",color='red',dashes=[2,2])
    axc.plot(t,vg1[:,0],label=r"guess",color='black',dashes=[1,1],alpha=0.5)
        
    axc.set_xlabel(r"\bf{Time of pathogen introduction} ${t_0}$",fontsize=22)

    #axc.subplot(nblignes,2,4)
    axc=ax[1,1]
    axdemiabsticks(axc)
    mettrelettreax(axc,'D',taille=50)
    axc.axis([0, 1.0, 0.0,0.4])
    axc.set_yticks((0.0, 0.1,0.2,0.3))
    axc.set_yticklabels((r'\bf{0}', r'\bf{0.1}', r'\bf{0.2}',r'\bf{0.3}'), color='k', size=20)

    if complet:
        axc.set_yticks((0.0, 0.1,0.2,0.3))
        axc.set_yticklabels((r'\bf{0}', r'\bf{0.1}',r'\bf{0.2}',r'\bf{0.3}'), color='k', size=20)
        axc.plot(t,z2[:,0],label=r"Exposed Human",color='black')
        axc.plot(t,z2[:,1],label=r"Infectious Human",color='black',dashes=[2,2])
        axc.plot(t,z2[:,2],label=r"Exposed Vector",color='red')
        axc.plot(t,z2[:,3],label=r"Infectious Vector",color='red',dashes=[2,2])
    else:
        axc.plot(t,z2[:,0],label=r"Exposed Human",color='black')
    axc.plot(t,vg2[:,0],label=r"guess",color='black',dashes=[1,1],alpha=0.5)
    axc.set_xlabel(r"\bf{Time of pathogen introduction} ${t_0}$",fontsize=22)
    def fpe2(x):
        return(z2[int(x*len(z1)),0])
    trueawinter=trouvepremierzero(fpe2,0.0,awinter,tolerance=4e-3)
    maxguess2=(vg2[:,0]).max()
    print("trueawinter=",trueawinter,"maxguess2",maxguess2)
    a1,b1=trouvepremieretdernierzero(fpe2,0.02,0.98,tolerance=5e-3)
    axbandegris(axc,awinter,bwinter)
    axbandegris(axc,a1,awinter,couleur=cgris,alpha=0.5)
  
    nomfig="zikaemergence.pdf"
    if complet:
        nomfig="complet"+nomfig
    plt.savefig(nomfig,dpi=300,bbox_inches='tight',pad_inches=0.2)

Lacolormap='Greys'
Lacolormarker='cyan'


################################ Mardi 19 mars 2019
def stepnaivevsoptimal(gamma=0.3,lzero=3.0/0.7,tunmax=1.0,T=70,nbpt=100,C=0.2,beamer=False):
    r"on fixe a priori une grande periode"
    #couleurcontrole='lightblue'
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def la(t):
        if (t< 1-gamma):
            return(lzero)
        else:
            return(0.0)
    def La(t):
        if (t< 1-gamma):
            return(lzero*t)
        else:
            return(lzero*(1-gamma))
    def phi(t):
        return(La(t) -Mu(t))

    pla=periodise(la)
    ############################## on definit la fonction optimale pour le cas step
    
    tdso=1-gamma-C*lzero/(lzero-1)
    rhoms=C/(1-gamma -tdso)
    print("tdso=",tdso)
    def lasopt(t):
        if (t <=tdso):
            return(lzero)
        elif (t<= 1-gamma):
            return(1)
        else:
            return(0.0)
    plasopt=periodise(lasopt)

    ###################" puis on definit la fonction naive
    rhomnaive=C/(1-gamma)
    def lasnaive(t):
        if (t<=1-gamma):
            return(lzero*(1-rhomnaive))
        else:
            return(0.0)
            
    plasnaive=periodise(lasnaive)
    ############# la on va tracer les courbes    
    et=np.linspace(0,2,200)
    t=np.linspace(0,1,100)
    #plt.rcParams['figure.figsize'] = [16, 10]
    if (beamer):
        fig, ax = plt.subplots(figsize=[16, 10])
    else:
        fig, ax = plt.subplots(figsize=[10, 16])
        

    ax.set_xlabel('time')
    ax.set_ylabel('rates')
    ax.plot(et,[plasnaive(s) for s in et],label=r"$\lambda_{naive}$",color='lightgray')
    ax.plot(et,[plasopt(s) for s in et],label=r"$\lambda_{opt}$",color=couleurcontrole)
    ax.plot(et,[pla(s) for s in et],label=r"$\lambda$",color='black')
    rectopt=Rectangle((tdso,lzero*(1-rhoms)),width=1-gamma-tdso,height= lzero*rhoms,color=couleurcontrole,alpha=0.5,fill=True)
    ax.add_patch(rectopt)
    rectnaive=Rectangle((0.0,lzero*(1-rhomnaive)),width=1-gamma,height= lzero*rhomnaive,color='lightgrey',alpha=0.5,fill=True)
    ax.add_patch(rectnaive)

    ax.legend()
    ax.plot(et,[1.0 for s in et],label=r"$\mu(t)$",color='black',dashes=[6,2])
    #plt.axis('scaled')
    ax.text(0.05,1.1,r"$\mu(t)$",fontsize=22)
    ax.text(0.05,la(0.0)+0.1,r"$\lambda(t)$",fontsize=22,color='black')

    
    plt.savefig("stepnaivevsoptimal.pdf",dpi=300)



def trouvecontour(z,tun,rho,disc=0.0001):
    w=np.where(z<=z.min()+disc)
    lesx=w[0]
    lesy=w[1]
    xc=lesx[0]
    yc=lesy[0]
    lec=[(rho[yc],tun[xc])]
    for i in range(1,len(lesx)):
        if (lesx[i]> xc):
            xc=lesx[i]
            yc=lesy[i]
            lec.append((rho[yc],tun[xc]))

    return(lec)


########################## Mercredi 20 mars 2020 : c'est le printemps
def explaincontrolstrategy(gamma=0.3,lzero=3.0,beamer=False,tun=0.0,tdeux=0.8,C=0.2):
    r"Explication de la strategie de controle avec la courbe rho et la courbe lambda_rho"
    #couleurcontrole='lightblue'
    def mu(t):
        return(1.0)
    def la(t):
        if (t< 1-gamma):
            return(lzero)
        else:
            return(0.0)
    def rho(t):
        if ((t>= tun) and (t<= tdeux)):
            return(rhom)
        else:
            return(0.0)
    def larho(t):
        return(la(t)*(1-rho(t)))
    
    #on definit les fonctions corespondant au controle optimal pour la sinusoide
    rhom=C/(tdeux -tun)
    assert(rhom <=1)

    pla=periodise(la)
    pmu=periodise(mu)
    plarho=periodise(larho)
    prho=periodise(rho)

 
    ############# la on va tracer les courbes    
    et=np.linspace(0,2,200)
    t=np.linspace(0,1,100)
    #plt.rcParams['figure.figsize'] = [16, 10]
    if (beamer):
        f,ax=plt.subplots(nrows=2, ncols=1, figsize=[16, 11])
    else:
        f,ax=plt.subplots(nrows=2, ncols=1, figsize=[11, 16])
        

    
    plt.subplot(2,1,1)
    plt.ylabel(r"$\rho(t)$",fontsize=22)
    plt.plot(et,[prho(s) for s in et],color='black')
    plt.axis([0.0, 2.0,0.0, 1.2*rhom])
    h=0.05
    plt.text(h/2,rhom+h/2,r"$\rho_M$",fontsize=22)
    plt.text(tun-2*h,h/2,r"$t_1$",fontsize=22)
    plt.text(tdeux+h,h/2,r"$t_2$",fontsize=22)

    
    plt.subplot(2,1,2)
    plt.ylabel(r"$\lambda_\rho(t)$",fontsize=22)
    plt.axis([0.0, 2.0,0.0, 1.2*lzero])

    plt.plot(et,[pla(s) for s in et],color='black',label=r"$\lambda(t)$")
    plt.plot(et,[plarho(s) for s in et],color=couleurcontrole,label=r"$\lambda_\rho(t)$")
    plt.legend(fontsize=16)
 

    plt.savefig("explainstepcontrolstrategy.pdf",dpi=300)

##################################################################
########## Mercredi 19 avril 2019 : winter coming pour le cas step

def winteriscoming(gamma=0.3,lzero=3.0,lT=[0.2,1,5,20,50,100],lzmin=0.5,pgr=True,beamer=True):
    #cgris='lightgrey'
    def mu(t):
        return(1.0)
    def Mu(t):
        return(t)
    def laoff(t):
        if (t< 1-gamma):
            return(lzero)
        else:
            return(lzmin)
    def Laoff(t):
        if (t< 1-gamma):
            return((lzero)*t)
        else:
            return(lzero*(1-gamma)+lzmin*(t-(1-gamma)))
    def la(t):
        if (t< 1-gamma):
            return(lzero)
        else:
            return(0.0)
    def La(t):
        if (t< 1-gamma):
            return((lzero)*t)
        else:
            return(lzero*(1-gamma))
    def phi(t):
        return(La(t) -Mu(t))

    def phioff(t):
        return(Laoff(t) -Mu(t))

    def grise():
        bandegris(1-gamma, 1.0)
        bandegris(1+1-gamma,2.0)
    def axgrise(ax):
        axbandegris(ax,1-gamma, 1.0)
        axbandegris(ax,1+1-gamma, 1+1.0)

    def bandegrisbarre(a,b):
        plt.axvspan(a, b, color='lightgray', alpha=0.5, lw=0,hatch='///')
    def grisebarre():
        bandegrisbarre(1-gamma, 1.0)
        bandegrisbarre(1+1-gamma,2.0)

    plt.rc('lines', linewidth=3.5, color='b')

    pla=periodise(la)
    pmu=periodise(mu)
    plaoff=periodise(laoff)
    
    et=np.linspace(0,2,200)
    t=np.linspace(0,1,100)
    #plt.rcParams['figure.figsize'] = [16, 10]
    nblignes=3 if pgr else 2
    if (beamer):
        f,ax=plt.subplots(nrows=nblignes, ncols=2, figsize=[16,13]) #pour les graphiques beamer
    else:
        f,ax=plt.subplots(nrows=nblignes, ncols=2, figsize=[10,16]) #pour les graphiques beamer
    #f.tight_layout()
    f.subplots_adjust(top=0.9,bottom=0.1,left=0.125,right=0.9,hspace=0.5)
    
    #plt.subplot(nblignes,2,1)
    axc=ax[0,0]
    axabsticks(axc)
    axgrise(axc)
    mettrelettreax(axc,'A',taille=50,gauche=0.0)
    axc.set_ylabel(r"\bf{Birth and Death rates}",fontsize=20)  
    vpla=np.array([pla(s) for s in et])
    vplaoff=np.array([plaoff(s) for s in et])
    vmax=max(vpla.max(),vplaoff.max())
    axc.plot(et,vplaoff,label=r"$\lambda(t)$",color='black')
    axc.plot(et,[pmu(s) for s in et],label=r"$\mu(t)$",color='black',dashes=[1,1])
    axc.axis([0.0, 2.0,0.0, 3.5]) #par defaut dans matplotlib sans seaborn
    axc.text(0.05,1.1,r"$\mu(t)$",fontsize=22)
    axc.text(0.05,laoff(0.0)+0.1,r"$\lambda(t)$",fontsize=22)
    axc.set_yticks((0,1,2,3))
    axc.set_yticklabels((r"\bf{0}",r"\bf{1}",r"\bf{2}",r"\bf{3}"))
    
    #plt.subplot(nblignes,2,2)
    axc=ax[0,1]
    mettrelettreax(axc,'B',taille=50)
    axabsticks(axc)
    axgrise(axc)
    axc.plot(et,vpla,label=r"$\lambda(t)$",color='black')
    axc.plot(et,[pmu(s) for s in et],label=r"$\mu(t)$",color='black',dashes=[1,1])
    axc.axis([0, 2.0, 0.0, 3.5]) #par defaut dans matplotlib sans seaborn
    axc.set_yticks((0,1,2,3))
    axc.set_yticklabels((r"\bf{0}",r"\bf{1}",r"\bf{2}",r"\bf{3}"))

    if (pgr):
        ##########################################
        ##### courbe phi dans le cas step
        ##################################
        #plt.subplot(nblignes,2,4)
        axc=ax[1,1]
        mettrelettreax(axc,'D',taille=50)
        axabsticks(axc)
        axgrise(axc)
        axc.set_yticks((0,1,2))
        axc.set_yticklabels((r"\bf{0}",r"\bf{1}",r"\bf{2}"))
        pphi=periodiseetintegre(phi)
        pphioff=periodiseetintegre(phioff)
        vpphi=np.array([pphi(s) for s in et])
        vpphioff=np.array([pphioff(s) for s in et])
        vphimax=max(vpphi.max(),vpphioff.max())
        axc.axis([0, 2.0, 0.0, 1.1*vphimax]) #par defaut dans matplotlib sans seaborn
        axc.plot(et,vpphi,label=r"$\varphi(t)$",color='black')
        tstar=(phi(1))/(lzero-1)
        minloc=phi(1)
        #la ligne pointillee horizontale depuis le minimum local
        abst=np.linspace(tstar,1,100)
        #je la vire le 28 mai 2019
        #line=plt.plot(abst,np.full_like(a=abst,fill_value=minloc),dashes=[6,2],color='black')[0]

        h=0.1

        def bgclair(ax):
            #enfin la bande gris clair au lieu de la fleche winter
            axbandegris(ax,tstar,1-gamma,alpha=0.5)
            axbandegris(ax,1+tstar,1+1-gamma,alpha=0.5)
        bgclair(axc)
        hep=0.04
        axc.arrow(1.0-0.02, 0.72, -(1.0 -tstar)+0.05,0.0, color='black', head_length = 0.06,linewidth=7 ,head_width = 0.09, length_includes_head = True,shape='full')
        axc.arrow(1.0+1.0-0.02, 1.48, -(1.0 -tstar)+0.05,0.0, color='black', head_length = 0.06,linewidth=7 ,head_width = 0.09, length_includes_head = True,shape='full')
        print("tstar=",tstar)


        ################################################################
        # on trace la courbe phi dans le cas step avec offset
        #plt.subplot(nblignes,2,3)
        axc=ax[1,0]
        mettrelettreax(axc,'C',taille=50,gauche=0.0)
     
        axabsticks(axc)
        axgrise(axc)
        axc.set_yticks((0,1,2))
        axc.set_yticklabels((r"\bf{0}",r"\bf{1}",r"\bf{2}"))

        axc.set_ylabel(r"\bf{Integrated growth rate}",fontsize=18)
        axc.text(0.2,0.7,r"$\varphi(t)$")
        axc.axis([0, 2.0, 0.0, 1.1*vphimax]) #par defaut dans matplotlib sans seaborn
        #plt.title(r"\bf{Integrated growth rate}",fontsize=22)
        axc.plot(et,vpphioff,label=r"$\varphi(t)$",color='black')


    #### proba emergence cas step avec offset
    if (pgr):
        #plt.subplot(nblignes,2,5)
        axc=ax[2,0]
    else:
        #plt.subplot(nblignes,2,3)
        axc=ax[1,0]
    mettrelettreax(axc,'E',taille=50,gauche=0.0)
    axgrise(axc)
    axabsticks(axc)
    axc.set_yticks((0,0.2,0.4,0.6,0.8))
    axc.set_yticklabels((r"\bf{0}",r"\bf{0.2}",r"\bf{0.4}",r"\bf{0.6}",r"\bf{0.8}"))
    axc.axis([0, 2.0, 0.0, 0.82]) #par defaut dans matplotlib sans seaborn
    
    axc.set_xlabel(r"\bf{Time of pathogen introduction} ${t_0}$",fontsize=22)
    axc.set_ylabel(r"\bf{Emergence probability}",fontsize=20)
    axc.text(0.1,0.63,r"$p_e(t_0T,T)$",color='k')
    vpemax=0.0
    cs=CubicSpline(np.array([0.0,10,100]),np.array([0.2,0.5,0.9]))

    for T in lT:
        pem=nncalculepe(laoff,Laoff,mu,Mu,T)
        ppem=periodise(pem)
        valppem=np.array([ppem(s) for s in et])
        axc.plot(et,valppem,label=r"T="+str(T),color=plt.cm.YlOrRd(cs(T)))
        vpemax=max(vpemax,valppem.max())

    def offguess(t):
        if (laoff(t)>=mu(t)):
            return(1-(mu(t)/laoff(t)))
        else:
            return(0.0)
    
    poffg=periodise(offguess)
    axc.plot(et,[poffg(s) for s in et],color='black',dashes=[6,2])
    legend = axc.legend(fontsize=12,loc='lower left')
    frame = legend.get_frame()
    frame.set_linewidth(0.0)

    #### proba emergence cas step
    if (pgr):
        #plt.subplot(nblignes,2,6)
        axc=ax[2,1]
    else:
        #plt.subplot(nblignes,2,4)
        axc=ax[1,1]
    mettrelettreax(axc,'F',taille=50)
    axabsticks(axc)
    axgrise(axc)
    axc.set_yticks((0,0.2,0.4,0.6,0.8))
    axc.set_yticklabels((r"\bf{0}",r"\bf{0.2}",r"\bf{0.4}",r"\bf{0.6}",r"\bf{0.8}"))
    
    axc.axis([0, 2.0, 0.0, 0.82]) #par defaut dans matplotlib sans seaborn
    axc.set_xlabel(r"\bf{Time of pathogen introduction} ${t_0}$",fontsize=22)

    for T in lT:
        pem=nncalculepe(la,La,mu,Mu,T)
        ppem=periodise(pem)
        color=plt.cm.YlOrRd(cs(T))
        axc.plot(et,[ppem(s) for s in et],label=r"T="+str(T),color=color)

    bgclair(axc)
    def stepguess(t):
        if (la(t)>=mu(t)):
            return(1-(mu(t)/la(t)))
        else:
            return(0.0)
    
    psg=periodise(stepguess)
    axc.plot(et,[psg(s) for s in et],label=r"guess($t_0$)",color='black',dashes=[6,2])

    if (pgr):
        plt.savefig("winteriscoming.pdf",dpi=300,bbox_inches='tight')
    else:
        plt.savefig("winteriscomingsansphi.pdf",dpi=300,bbox_inches='tight')


def absticks():
        plt.xticks((0.0,0.5,1.0,1.5,2.0),(r"\bf{0}",r"\bf{0.5}",r"\bf{1.0}",r"\bf{1.5}",r"\bf{2.0}"))
def axabsticks(ax):
        ax.set_xticks((0.0,0.5,1.0,1.5,2.0))
        ax.set_xticklabels((r"\bf{0}",r"\bf{0.5}",r"\bf{1.0}",r"\bf{1.5}",r"\bf{2.0}"))
  
def demiabsticks():
        plt.xticks((0.0,0.2,0.4,0.6,0.8,1.0),(r"\bf{0}",r"\bf{0.2}",r"\bf{0.4}",r"\bf{0.6}",r"\bf{0.8}",r"\bf{1}"))
def axdemiabsticks(ax):
        ax.set_xticks((0.0,0.2,0.4,0.6,0.8,1.0))
        ax.set_xticklabels((r"\bf{0}",r"\bf{0.2}",r"\bf{0.4}",r"\bf{0.6}",r"\bf{0.8}",r"\bf{1}"))



################ Mardi 4 juin : illustrons sur l'exemple de sylvain laconvergence de l'ode de Bacaer vers la prob d'emergence

def tbacasyl(T=100,lzero=0.2,muzero=0.1,gamma=0.3,nbperiodes=3):
    def mu(t):
        return(muzero)
    def Mu(t):
        return(muzero*t)
   #il faut ajuster le lzerosin pour que les processus aient le meme rzero

    def lastep(t):
        if (t< 1-gamma):
            return(lzero)
        else:
            return(0.0)
    def Lastep(t):
        if (t< 1-gamma):
            return(lzero*t)
        else:
            return(lzero*(1-gamma))
    def guess(t):
        a=lastep(t)/mu(t)
        if (a>1):
            return(1-(1/a))
        else:
            return(0.0)
    plastep=periodise(lastep)
    pmu=periodise(mu)
    #nbpts=nb*tau
    Lambda=[[plastep]]
    VMu=[pmu]
    #z=vfbaca(Lambda,VMu,T=T,tv=False)
    fig, ax = plt.subplots(2,1,figsize=[16, 12])
    plt.tight_layout()
    
    plt.subplot(2,1,1)
    z=nvfbaca(Lambda,VMu,T=T,tv=False,nbtau=nbperiodes)#fait en plus un plot
    z=z.reshape(len(z),)
    pem=nncalculepe(lastep,Lastep,mu,Mu,T)
    t=np.linspace(0.0,1.0,len(z))
    valpem=np.array([pem(s) for s in t])
    vg=np.array([guess(s) for s in t])
    
    plt.subplot(2,1,2)
    #plt.title("Square wave  birth rate")
    plt.ylabel(r"${p_e(t_0 T,T)}$",fontsize=22)
    plt.plot(t,z,label="simulation de Bacaer " +str(nbperiodes)+" periodes",color=couleurcontrole)
    plt.plot(t,valpem,label="calcul integral exact",dashes=[3,3],color='black')
    plt.xlabel(r"\textbf{Introduction time} ${t_0}$",fontsize=22)
    #pas besoin de mettre le guess je pense 
    #plt.plot(t,vg,label=r"guess : $(1-\lambda/\mu)^+$",linestyle=':')
    #plt.legend()
    plt.savefig("approximationbacaer1d.pdf",dpi=300)
    print("difference max",np.abs(valpem-z).max())
    #on n'a pas zero car les tableaux n'ont pas meme dimension
    return((z,valpem))


def nvfbaca(Lambda,Mu,T,tv=False,nbtau=10,toute=True):
    r""" la on essaie de plotter toute la solution de l'ode de bacaer"""
    tau=nbtau*T
    nbpts=10*tau
    d=len(Mu)
    zzero=np.array([1.0 for i in range(d)])
    
    def msisi(x,t):
        #s=tau-(t/T)
        s=(tau-t)/T
        return(np.array([-x[i]*Mu[i](s) + (1-x[i])*((np.array([Lambda[i][j](s)*x[j] for j in range(d)])).sum()) for i in range(d)]))

    timeint=np.arange(0,tau+1/nbpts,tau/nbpts)
    z=np.array(odeint(msisi,zzero,timeint))
    if toute:
        plt.ylabel(r"${Y^{(\tau)}}$",fontsize=22)
        plt.plot(timeint,z,label=r"$Y^{(\tau)}$",color=couleurcontrole)
        plt.xlabel(r"\textbf{Approximation time : } ${s=\tau-t_0}$",fontsize=22)
    i=int(T*nbpts/tau)
    #print("T=",T,"tau=",tau,"z.shape",z.shape,"i=",i)
    res=z[-i:]
    if tv:
        plt.plot(timeint[-2*i:],z[-2*i:],label="z")
        plt.legend()
    return(res[::-1])

