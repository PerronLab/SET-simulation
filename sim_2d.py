# written by Erick Daniel Ochoa, CSUSM Fall '18
# master equation simulation of electron transport through a quantum dot with spin states
import numpy as np
import matplotlib.pyplot as plt
import time
start  = time.time() 

########################################CONSTANTS#######################################
# 1. declare variables
e = 1.602e-19 # electron charge [C]
kb = 8.617e-5 # Boltzmanns constant [eV/K]
J2eV = 6.2415e18 # joule to meV conversion [J/eV]
tL = tR = 0.04e9 # tunneling rates {find from lit!}
gammaL = 2*np.pi*tL**2 # tunneling rate onto dot from left
gammaR = 2*np.pi*tR**2 # tunneling rate onto dot from right

dL = dR = 125e-9 # distance to dot [m]
bohrmag = 9.274e-24 # bohr-magneton [J/K]
######################################PARAMETERS#######################################
Cs = 2.5e-18 # source capacitance [C]
Cg = 2e-18 # gate capacitance [C]
Cd = 1e-18 # drain capacitance [C]
Clg = 2e-18 # left gate capacitance [C]
Crg = 2e-18 # right gate capacitance [C]
Ctot = Cs + Cg + Cd + Clg + Crg # total dot capacitance [C]
Eq = 0
Ec = e**2/Ctot # charging energy [J]
T = .50 # temperature [K]
beta  = 1/(T*kb)
dsteps = 300 # number of drain voltage steps
psteps = 200 # number of plunger voltage steps

Vs = 0 # source voltage [V]
Vdstart = -0.0175 # starting drain voltage [V]
Vdstop = 0.025 # stopping drain voltage [V]
d = np.linspace(Vdstart,Vdstop,num= dsteps) # 1D drain voltage array
Vd = np.zeros(shape=(psteps,dsteps)) # 2D drain voltage array

Vpstart = -0.02 # starting plunger gate voltage [V]
Vpstop = 0.2 # stopping plunger gate voltage [V]
p = np.linspace(Vpstart,Vpstop,num= psteps) # 1D plunger voltage array
Vp = np.zeros(shape=(psteps,dsteps)) # 2D plunger voltage array

B  = 10 # magnetic field in T
zeeman  = bohrmag*B # Zeeman energy [J]
#####################################MAIN LOOP######################################### 
# Possible energies for a two energy level system (N=2)
Eo = [] # ground state energy [J] (N=0)
E1u = [] # 1st energy level for spin up electrons [J] (N=1)
E1d = [] # 1st energy level for spin down electrons [J] (N=1)
E2 = [] # 2nd energy level [J] (N=2)

fermi1 = [] # fermi function for N=1
fermi2 =[] # fermi function for N=2


I = np.zeros(shape=(psteps,dsteps)) # current [A]
P = [] # total probability
Po = np.zeros(dsteps) # probability of zeroth state occupied
P1u = np.zeros(dsteps) # probabilty of first spin-up state occupied
P1d = np.zeros(dsteps) # probability of first spin-down state occupied
P2 = np.zeros(dsteps) # probability of second state occupied
count1 = 0
count2 = 0

# iterate over Vd
for val in d:
    vdrain = val
    
    for val in p:
        # 2. calculate charge state energies
  
        Vd[count1,count2] = vdrain 
        Vp[count1,count2] = val 
        
        # N = 0
        Eonow = ((-e*(0)+ Cs*Vs + Cd*vdrain + Cg*val)**2/(2*Ctot))*J2eV
        
        # N = 1
        E1unow = (((-e*(1)+ Cs*Vs + Cd*vdrain + Cg*val)**2/(2*Ctot))+zeeman)*J2eV
        E1dnow = (((-e*(1)+ Cs*Vs + Cd*vdrain + Cg*val)**2/(2*Ctot))-zeeman)*J2eV
     
        # N = 2
        E2now = (((-e*(2)+ Cs*Vs + Cd*vdrain + Cg*val)**2/(2*Ctot))+2*Eq)*J2eV

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

        # all possible tunneling transitions
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

        # current = all possible ways on state - all possible ways off state
        on1 = Po[count1]*gammaL*(fermi1dL + fermi1uL) # possible ways on first state 
        on2 = P1d[count1]*gammaL*fermi2dL + P1u[count1]*gammaL*fermi2uL # possible ways on second state
        off1 = P1d[count1]*gammaL*(1-fermi1dL) + P1u[count1]*gammaL*(1-fermi1uL) # possible ways off first state
        off2 = P2[count1]*gammaL*((1-fermi2dL) + (1-fermi2uL)) # possible ways off second state

        Inow = (-e)*(on1 + on2 - off1 - off2)
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
