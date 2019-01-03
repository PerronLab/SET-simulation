# ULTRA-LOW TEMPERATURE INVESTIGATION OF SILICON QUANTUM DOTS
# master equation simulation of electron transport
# through a quantum dot
import numpy as np
import matplotlib.pyplot as plt
import time
start  = time.time() 

########################################CONSTANTS#######################################
# 1. declare variables
e = 1.602e-19
kb = 8.617e-5
J2meV = 6.2415e18 # joule to meV conversion
meV2J = 1.602e-22
tL = tR = 0.04e9
W = 0.001

gammaR = 1.6e6
gammaL = 2*gammaR
#gammaL = 2*np.pi*tL**2
#gammaR = 2*np.pi*tR**2

dL = dR = 125e-9
bohrmag = 9.274e-24 # in Joules per Tesla
######################################PARAMETERS#######################################
Cs = 2.5e-18
Cg = 1e-18
Cd = 2e-18
Clg = 2e-18
Crg = 2e-18
# Ctot = Cs + Cg + Cd + Clg + Crg
Eq = 0
# Ec = e**2/Ctot
Ec = 2.4*meV2J
Ctot = e**2/Ec
T = 50e-3
beta  = 1/(T*kb)
dsteps = 300
psteps = 200

Vs = -0.05
Vdstart = -0.2
Vdstop = 0.2
d = np.linspace(Vdstart,Vdstop,num= dsteps)
Vd = np.zeros(shape=(psteps,dsteps))

Vpstart = -5
Vpstop = -2
p = np.linspace(Vpstart,Vpstop,num= psteps)
# Vp = np.zeros(shape=(psteps,dsteps))
Vp  = np.zeros(shape=(psteps,dsteps))

B  = 10
zeeman  = bohrmag*B
orbital = 1.1*meV2J
####################################################################################### 
Eo = []
E1g = []
E1e = []
mu1 = []
mu2 = []
fermi1L = []
fermi2L = []
fermi1R = []
fermi2R = []

I = np.zeros(shape=(psteps,dsteps))
P = []
Po = np.zeros(psteps)
P1g = np.zeros(psteps)
P1e = np.zeros(psteps)

count1 = 0
count2 = 0

# iterate over Vd
for val in d:
    vdrain = val

    for val in p:
        # 2. calculate charge state energies
        #print count1, count2
        # vdrain = 2

        Vd[count1,count2] = vdrain
        # Vd[count1] = vdrain
        Vp[count1,count2] = val
        # Vd[count1] = vdrain
        
        
        # N = 0
        Eonow = ((-e*(0)+ Cs*Vs + Cd*vdrain + Cg*val)**2/(2*Ctot))*J2meV
        #Eo.append(Eonow)

        # N = 1
        E1gnow = ((-e*(1)+ Cs*Vs + Cd*vdrain + Cg*val)**2/(2*Ctot))*J2meV
        #E1g.append(E1gnow)
        #mu1now = E1gnow - Eonow
        #mu1.append(mu1now)
        E1enow = (((-e*(1)+ Cs*Vs + Cd*vdrain + Cg*val)**2/(2*Ctot))+orbital)*J2meV
        #E1e.append(E1enow)
        #mu2now = E1enow - Eonow
        #mu2.append(mu2now)

        # 3. calculate chemical potentials and Fermi functions
        muL = -Vs
        muR = -vdrain

        fermi1gL = 1/(np.exp(beta*(E1gnow-Eonow-muL))+1)
        #fermi1L.append(fermi1gL)

        fermi1eL = 1/(np.exp(beta*(E1enow-Eonow-muL))+1)
        #fermi2L.append(fermi1eL)
        
        fermi1gR = 1/(np.exp(beta*(E1gnow-Eonow-muR))+1)
        #fermi1R.append(fermi1gR)
        
        fermi1eR = 1/(np.exp(beta*(E1enow-Eonow-muR))+1)
        #fermi2R.append(fermi1eL)

        # 4. calculate tunneling rates
        gamma1g0 = gammaL*fermi1gL + gammaR*fermi1gR
        gamma1e0 = gammaL*fermi1eL + gammaR*fermi1eR 
        gamma01g = gammaL*(1-fermi1gL) + gammaR*(1-fermi1gR) 
        gamma01e = gammaL*(1-fermi1eL) + gammaR*(1-fermi1eR) 

        # 5. define A, P, and C matrices and solve for P
        A = np.matrix([
            [(gamma1e0+gamma1g0), -gamma01g, -gamma01e],
            [-gamma1g0, gamma01g, -W],
            [1, 1, 1]])
        C = np.array([[0],[0],[1]])
        Ainv = A.getI()

        Pnow = Ainv*C
        Po[count1] = Pnow[0]
        P1g[count1] = Pnow[1]
        P1e[count1] = Pnow[2]

        on1 = Po[count1]*gammaR*(fermi1gR + fermi1eR)
        off1 = P1g[count1]*gammaR*(1-fermi1gR) + P1e[count1]*gammaR*(1-fermi1eR)

        """ on1 = Po[count1]*gammaR*(fermi1gL + fermi1eL)
        off1 = P1g[count1]*gammaL*(1-fermi1gL) + P1e[count1]*gammaL*(1-fermi1eL) """

        Inow = (-e)*(on1 - off1)
        I[count1,count2] = Inow
        count1 +=1
        if count1 > psteps-1:
            count1 = 0
    count2 +=1
    if count2 > dsteps-1:
        count2 = 0
# 6. plot I vs. Vd
# print Vd.size, Vp.size, I.size
#######################################FOR dI/dV########################################
# plot dI/dV

# for troubleshooting
""" plt.subplot(221)
plt.title('T = '+str(T)+' K, W = '+str(W)+' s^-1, Vd = Vs = '+str(Vs)+'V')
plt.plot(Vp,Eo,label='Eo')
plt.plot(Vp,E1g,label='E1g')
plt.plot(Vp,E1e,label='E1e')
plt.ylabel('Energy (meV)')
plt.legend()

plt.subplot(222) 
plt.plot(Vp,mu1,label='mu 1g')
plt.plot(Vp,mu2,label='mu 1e')
plt.legend()

plt.subplot(223)
plt.plot(Vp,fermi1L,label='fermi 1gL')
plt.plot(Vp,fermi2L,label='fermi 1eL')
plt.plot(Vp,fermi1R,label='fermi 1gR')
plt.plot(Vp,fermi2R,label='fermi 1eR')
plt.xlabel('Vp (V)')
plt.ylabel('Probability')
plt.legend()

plt.subplot(224)
plt.plot(Vp,Po,label='Po')
plt.plot(Vp,P1g,label='P1g')
plt.plot(Vp,P1e,label='P1e')
plt.xlabel('Vp (V)')
plt.legend()
plt.show() """

# calculate the gradient
""" gradI = np.gradient(I) # returns a 2 2D arrays:
gradIrow = gradI[0] # 1.)   gradient by row
gradIcol = gradI[1] # 2.)   gradient by column """

# plot dI/dVd vs Vp and Vd
# plt.subplot(212)
""" plt.pcolormesh(Vp,Vd,I, cmap=plt.get_cmap('seismic'))
plt.xlabel('Vp (V)')
plt.ylabel('Vd (V)')
cbar = plt.colorbar()
cbar.set_label('dI/dVd (A)')
plt.show() """
#########################################################################################
#plt.subplot(211)
plt.pcolormesh(Vp,Vd,I, cmap=plt.get_cmap('RdBu'))
cbar = plt.colorbar()
plt.ylabel('Vd (V)')
cbar.set_label('IDC (A)')

end  = time.time()
print "runtime:",end - start, "seconds"
plt.show()