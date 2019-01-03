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
tL = tR = 0.04e9
gammaL = 2*np.pi*tL**2
gammaR = 2*np.pi*tR**2

dL = dR = 125e-9
bohrmag = 9.274e-24 # in Joules per Tesla
######################################PARAMETERS#######################################
Cs = 2.5e-18
Cg = 2e-18
Cd = 1e-18
Clg = 2e-18
Crg = 2e-18
Ctot = Cs + Cg + Cd + Clg + Crg
Eq = 0
Ec = e**2/Ctot
T = 50
beta  = 1/(T*kb)
dsteps = 300
psteps = 200

Vs = 0
Vdstart = -0.0175
Vdstop = 0.025
d = np.linspace(Vdstart,Vdstop,num= dsteps)
Vd = np.zeros(shape=(psteps,dsteps))

Vpstart = -0.02
Vpstop = 0.1
p = np.linspace(Vpstart,Vpstop,num= psteps)
Vp = np.zeros(shape=(psteps,dsteps))

B  = 10
zeeman  = bohrmag*B
####################################################################################### 
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

I = np.zeros(shape=(psteps,dsteps))
P = []
Po = np.zeros(dsteps)
P1u = np.zeros(dsteps)
P1d = np.zeros(dsteps)
P2 = np.zeros(dsteps)
count1 = 0
count2 = 0

# iterate over Vd
for val in d:
    vdrain = val
    
    for val in p:
        # 2. calculate charge state energies
        #print count1, count2
        Vd[count1,count2] = vdrain 
        Vp[count1,count2] = val 
        
        # N = 0
        Eonow = ((-e*(0)+ Cs*Vs + Cd*vdrain + Cg*val)**2/(2*Ctot))*J2meV
        #Eo.append(Eonow)
        # N = 1
        E1unow = (((-e*(1)+ Cs*Vs + Cd*vdrain + Cg*val)**2/(2*Ctot))+zeeman)*J2meV
        #E1u.append(E1unow)
        # mu1now = E1unow - Eonow
        # mu1.append(mu1now)
        E1dnow = (((-e*(1)+ Cs*Vs + Cd*vdrain + Cg*val)**2/(2*Ctot))-zeeman)*J2meV
        #E1d.append(E1dnow)
        # N = 2
        E2now = (((-e*(2)+ Cs*Vs + Cd*vdrain + Cg*val)**2/(2*Ctot))+2*Eq)*J2meV
        #E2.append(E2now)
        # mu2now = E2now - E1unow
        # mu2.append(mu2now)

        # 3. calculate chemical potentials and Fermi functions
        muL = -Vs
        muR = -vdrain

        fermi1uL = 1/(np.exp(beta*(E1unow-Eonow-muL))+1)
        fermi1dL = 1/(np.exp(beta*(E1dnow-Eonow-muL))+1)
        fermi2uL =  1/(np.exp(beta*(E2now-E1unow-muL))+1)
        fermi2dL =  1/(np.exp(beta*(E2now-E1dnow-muL))+1)

        fermi1uR = 1/(np.exp(beta*(E1unow-Eonow-muR))+1)
        fermi1dR = 1/(np.exp(beta*(E1dnow-Eonow-muR))+1)    
        fermi2uR =  1/(np.exp(beta*(E2now-E1unow-muR))+1)
        fermi2dR =  1/(np.exp(beta*(E2now-E1dnow-muR))+1)

        # 4. calculate tunneling rates

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
        Po[count1] = Pnow[0]
        P1u[count1] = Pnow[1]
        P1d[count1] = Pnow[2]
        P2[count1] = Pnow[3]

        # on1 = Po[count]*gamma1d0 + Po[count]*gamma1u0 + P2[count]*(gamma1u2+gamma1d2)
        # on0 = P1d[count]*gamma
        on1 = Po[count1]*gammaL*(fermi1dL + fermi1uL)
        on2 = P1d[count1]*gammaL*fermi2dL + P1u[count1]*gammaL*fermi2uL
        off1 = P1d[count1]*gammaL*(1-fermi1dL) + P1u[count1]*gammaL*(1-fermi1uL)
        off2 = P2[count1]*gammaL*((1-fermi2dL) + (1-fermi2uL))

        Inow = (-e)*(on1 + on2 - off1 - off2)
        # Inow = (-e)*(gammaL*fermi2dL*P1d[count] + gammaL*fermi2uL*P1u[count]
        # - gammaL*(1-fermi2uL)*P2[count] + gammaL*(1-fermi2dL)*P2[count]) 
        I[count1,count2] = Inow
        count1 +=1
        if count1 > psteps-1:
            count1 = 0
    count2 +=1
    if count2 > dsteps-1:
        count2 = 0
# 6. plot I vs. Vd
Esplit = round((E1unow - E1dnow)/2, 6)
#######################################FOR dI/dV########################################
# plot dI/dV

# calculate the gradient
gradI = np.gradient(I) # returns a 2 2D arrays:
gradIrow = gradI[0] # 1.)   gradient by row
gradIcol = gradI[1] # 2.)   gradient by column

# plot dI/dVd vs Vp and Vd
# plt.subplot(212)
plt.pcolormesh(Vp,Vd,gradIrow, cmap=plt.get_cmap('seismic'))
plt.xlabel('Vp (V)')
plt.ylabel('Vd (V)')
cbar = plt.colorbar()
cbar.set_label('dI/dVd (A)')
#########################################################################################
#plt.subplot(211)
""" plt.pcolormesh(Vp,Vd,I, cmap=plt.get_cmap('seismic'))
cbar = plt.colorbar()
plt.ylabel('Vd (V)')
cbar.set_label('IDC (A)')
plt.title('B= '+ str(B) +'T, T= ' + str(T) + 'K, Energy splitting: '
    + str(Esplit)+ ' meV' )

end  = time.time()
print "runtime:",end - start, "seconds" """
plt.show()