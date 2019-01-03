# ULTRA-LOW TEMPERATURE INVESTIGATION OF SILICON QUANTUM DOTS
# master equation simulation of electron transport
# through a quantum dot
import numpy as np
import matplotlib.pyplot as plt 

# 1. declare variables
Vs = 0
Vg = 0
Cs = 2.5e-18
Cg = 2e-18
Cd = 1e-18
Clg = 2e-18
Crg = 2e-18
Ctot = Cs + Cg + Cd + Clg + Crg
e = 1.602e-19

T = 20
kb = 8.617e-5
beta  = 1/(T*kb)
J2meV = 6.2415e18 # joule to meV conversion

nsteps = 1000

Vdstart = -0.1
Vdstop = 0.3
Vd = np.linspace(Vdstart,Vdstop,num= nsteps)

Eq = 0
Ec = e**2/Ctot
Eo = []
E1u = []
E1d = []
E2 = []
mu1 = []
mu2 = []
fermi1 = []
fermi2 =[]
fermi1 = []
fermi2 =[]


tL = tR = 0.04e9
dL = dR = 125e-9

I = []
P = []
Po = np.zeros(nsteps)
P1u = np.zeros(nsteps)
P1d = np.zeros(nsteps)
P2 = np.zeros(nsteps)
count = 0
# iterate over Vd
for val in Vd:
    # 2. calculate charge state energies
    # print energies 
    # N = 0
    Eonow = ((-e*(0)+ Cs*Vs + Cd*val + Cg*Vg)**2/(2*Ctot))*J2meV
    Eo.append(Eonow)
    # N = 1
    E1unow = ((-e*(1)+ Cs*Vs + Cd*val + Cg*Vg)**2/(2*Ctot)+Eq)*J2meV
    E1u.append(E1unow)
    mu1now = E1unow - Eonow
    mu1.append(mu1now)
    E1dnow = E1unow
    E1d.append(E1dnow)
    # N = 2
    E2now = (((-e*(2)+ Cs*Vs + Cd*val + Cg*Vg)**2/(2*Ctot))+2*Eq)*J2meV
    E2.append(E2now)
    mu2now = E2now - E1unow
    mu2.append(mu2now)

    # 3. calculate chemical potentials and Fermi functions
    muL = -Vs
    muR = -val

    fermi1uL = 1/(np.exp(beta*(E1unow-Eonow-muL))+1)
    fermi1dL = 1/(np.exp(beta*(E1dnow-Eonow-muL))+1)
    fermi2uL =  1/(np.exp(beta*(E2now-E1unow-muL))+1)
    fermi2dL =  1/(np.exp(beta*(E2now-E1dnow-muL))+1)

    fermi1uR = 1/(np.exp(beta*(E1unow-Eonow-muR))+1)
    fermi1dR = 1/(np.exp(beta*(E1dnow-Eonow-muR))+1)
    fermi2uR =  1/(np.exp(beta*(E2now-E1unow-muR))+1)
    fermi2dR =  1/(np.exp(beta*(E2now-E1dnow-muR))+1)
    fermi1.append(fermi1uR)
    fermi2.append(fermi2uR)

    # 4. calculate tunneling rates
    gammaL = 2*np.pi*tL**2
    gammaR = 2*np.pi*tR**2

    # tunnel right (use gammaR)
    gamma1u0 = gammaL*fermi1uL + gammaR*fermi1uR
    gamma1d0 = gammaL*fermi1dL + gammaR*fermi1dR 
    gamma01d = gammaL*(1-fermi1dL) + gammaR*(1-fermi1dR) 
    gamma01u = gammaL*(1-fermi1uL) + gammaR*(1-fermi1uR) 
    gamma21u = gammaL*(fermi2uL) + gammaR*(fermi2uR) 
    gamma21d = gammaL*(fermi2dL) + gammaR*(fermi2dR) 
    gamma1u2 = gammaL*(1-fermi2uL) + gammaR*(1-fermi2uR) 
    gamma1d2 = gammaL*(1-fermi2dL) + gammaR*(1-fermi2dR) 

    # 5. define A, P, and C matrices and solve for P
    A = np.matrix([
        [-(gamma1u0+gamma1d0), gamma01u, gamma01d, 0],
        [gamma1u0, -(gamma01u+gamma21u), 0, gamma1u2],
        [gamma1d0, 0, -(gamma01d+gamma21d), gamma1d2],
        [1, 1, 1, 1]])
    C = np.array([[0],[0],[0],[1]])
    Ainv = A.getI()

    Pnow = Ainv*C
    Po[count] = Pnow[0]
    P1u[count] = Pnow[1]
    P1d[count] = Pnow[2]
    P2[count] = Pnow[3]

    # on1 = Po[count]*gamma1d0 + Po[count]*gamma1u0 + P2[count]*(gamma1u2+gamma1d2)
    # on0 = P1d[count]*gamma
    on1 = Po[count]*gammaL*(fermi1dL + fermi1uL)
    on2 = P1d[count]*gammaL*fermi2dL + P1u[count]*gammaL*fermi2uL
    off1 = P1d[count]*gammaL*(1-fermi1dL) + P1u[count]*gammaL*(1-fermi1uL)
    off2 = P2[count]*gammaL*((1-fermi2dL) + (1-fermi2uL))

    Inow = (-e)*(on1 + on2 - off1 - off2)
    # Inow = (-e)*(gammaL*fermi2dL*P1d[count] + gammaL*fermi2uL*P1u[count]
    # - gammaL*(1-fermi2uL)*P2[count] + gammaL*(1-fermi2dL)*P2[count]) 
    I.append(Inow)
    count +=1

# 6. plot I vs. Vd

#################################Energies################################
""" plt.subplot(3,1,1)
plt.plot(Vd,Eo,label='E0')
plt.plot(Vd,E1u,label='E1u')
plt.plot(Vd,E1d,label='E1d')
plt.plot(Vd,E2,label='E2')
plt.ylabel('Energy (meV)')
plt.legend(loc='right') """

###############################chem. pot.'s##############################
""" plt.subplot(3,1,2)
plt.plot(Vd,mu1,label='mu1')
plt.plot(Vd,mu2,label='mu2')
plt.ylabel('chemical potential (meV)')
plt.legend()
plt.grid(True) """
#################################Fermi's#################################
""" plt.subplot(3,1,3)
plt.plot(Vd,fermi1,label='Fermi 1')
plt.plot(Vd,fermi2,label='Fermi 2')
plt.xlabel('Vd (V)')
plt.ylabel('nf')
plt.legend(loc='right')
plt.grid(True) """
##############################Probabilities##############################
plt.subplot(2,1,1)
plt.plot(Vd,I,'k-',label='I')
plt.title('single quantum dot simulation')
plt.ylabel('I (A)')
plt.legend()
plt.subplot(2,1,2)
plt.plot(Vd,Po,label='P0')
plt.plot(Vd,P1u,label='P1u')
plt.plot(Vd,P1d,label='P1d')
plt.plot(Vd,P2,label='P2')
plt.ylabel('Probability')
plt.legend()
########################################################################
plt.show() 