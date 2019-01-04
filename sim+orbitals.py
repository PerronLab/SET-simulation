# written by Erick Daniel Ochoa, CSUSM Fall '18
# master equation simulation of electron transport throught a quantum dot with a single ground and excited state
import numpy as np
import matplotlib.pyplot as plt
import time
start  = time.time() 

########################################CONSTANTS#######################################
# 1. declare variables
e = 1.602e-19 # electron charge [C]
kb = 8.617e-5 # Boltzmanns constant [eV/K]
J2eV = 6.2415e18 # joule to eV conversion [J/eV]
eV2J = 1.602e-22 # eV to joule [eV/J]
tL = tR = 0.04e9 # tunneling rates {find from lit!}
W = 0.001 # relaxation rate 

gammaR = 1.6e6 # tunnel rate onto dot from right
gammaL = 2*gammaR # tunnel rate onto dot from left 

dL = dR = 125e-9 # distance to dot [m]
bohrmag = 9.274e-24 # bohr-magneton [J/K]
######################################PARAMETERS#######################################
Cs = 2.5e-18 # source capacitance [C]
Cg = 1e-18 # gate capacitance [C]
Cd = 2e-18 # drain capacitance [C]
Clg = 2e-18 # left gate capacitance [C]
Crg = 2e-18 # right gate capacitance [C]
Eq = 0
Ec = 2.4*eV2J # charging energy [J]
Ctot = e**2/Ec # total dot capacitance [C]
T = 50e-3 # temperature [K]
beta  = 1/(T*kb)
dsteps = 300 # number of drain voltage steps
psteps = 200 # number of plunger voltage steps

Vs = -0.05 # source voltage [V]
Vdstart = -0.2 # starting drain voltage [V]
Vdstop = 0.2 # stopping drain voltage [V]
d = np.linspace(Vdstart,Vdstop,num= dsteps) # 1D drain voltage array
Vd = np.zeros(shape=(psteps,dsteps)) # 2D drain voltage array

Vpstart = -5 # starting plunger gate voltage [V]
Vpstop = -2 # stopping plunger gate voltage [V]
p = np.linspace(Vpstart,Vpstop,num= psteps) # 1D plunger voltage array
Vp  = np.zeros(shape=(psteps,dsteps)) # 2D plunger voltage array

B  = 10 # magnetic field [T]
zeeman  = bohrmag*B # Zeeman energy [J]
orbital = 1.1*eV2J # orbital energy splitting [J] 
####################################################################################### 
# Possible energies for one energy level system with an excited state
Eo = []  # ground state energy [J] (N=0)
E1g = [] # 1st ground state energy [J] (N=1)
E1e = [] # 1st excited state energy [J] (N=1)

fermi1L = [] # left N=1 Fermi function
fermi2L = [] # left N=2 Fermi function
fermi1R = [] # right N=1 Fermi function
fermi2R = [] # right N=2 Fermi function

I = np.zeros(shape=(psteps,dsteps)) # dot current [A]
P = [] # probability 
Po = np.zeros(psteps) # probablity of zeroth state occupied
P1g = np.zeros(psteps) # probabilty of first ground state occupied
P1e = np.zeros(psteps) # probabilty of first excited state occupied

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
        E1gnow = ((-e*(1)+ Cs*Vs + Cd*vdrain + Cg*val)**2/(2*Ctot))*J2eV

        E1enow = (((-e*(1)+ Cs*Vs + Cd*vdrain + Cg*val)**2/(2*Ctot))+orbital)*J2eV

        # 3. calculate chemical potentials and Fermi functions
        muL = -Vs
        muR = -vdrain

        fermi1gL = 1/(np.exp(beta*(E1gnow-Eonow-muL))+1)
        fermi1eL = 1/(np.exp(beta*(E1enow-Eonow-muL))+1)
        fermi1gR = 1/(np.exp(beta*(E1gnow-Eonow-muR))+1)
        fermi1eR = 1/(np.exp(beta*(E1enow-Eonow-muR))+1)

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

        # current = all possible ways on state - all possible ways off state
        on1 = Po[count1]*gammaR*(fermi1gR + fermi1eR) # possible ways onto first state
        off1 = P1g[count1]*gammaR*(1-fermi1gR) + P1e[count1]*gammaR*(1-fermi1eR) # possible ways off first state
        
        Inow = (-e)*(on1 - off1)
        I[count1,count2] = Inow
        count1 +=1
        if count1 > psteps-1:
            count1 = 0
    count2 +=1
    if count2 > dsteps-1:
        count2 = 0
# 6. plot I vs. Vd
#######################################FOR dI/dV########################################
# plot dI/dV

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
print( "runtime:",end - start, "seconds")
plt.show()