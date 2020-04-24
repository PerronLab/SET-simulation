''' Attempting to simulate transport through a single quantum dot
using master equation approach. This simulation includes only charge
states. 
'''
import matplotlib
import matplotlib.pyplot as plt
#from matplotlib.mlab import griddata
from scipy.special import expit
import numpy as np


## constants
q = 1.602e-19; 		# fundamental charge (C)
kb = 1.381e-23; 	# Boltzmann constant (J/K)
kbeV = 8.6173e-5; 	# Boltzmann constant (eV/K)
h = 4.1357e-15;		# Plank's constant (eV s)
hbar = 6.582e-16;	# Reduced Plank's constant (h/2pi) (eV s)
# Physical parameters
q0 = 0; 		# Background charge assumed to be 0
###########################################################################
# Details of the simulation parameters
temp = 0.25;		# Temperature (of electrons in reservoirs) (K)
temps = [0.05, 0.5, 2]    # temperatures to run simulation at
LegTxt = ["" for i in temps]
j = 0
for i in temps:
    LegTxt[j] = f"T = {i} K"
#    LegTxt[j] = text
    print(LegTxt[j])
    j = j+1
Betas = [1/(temp*kbeV) for temp in temps]
###########################################################################

# Device parameters, simple 3 gates - 2 barrier (left,right) and one plunger and source drain leads
# Capacitances
m2 = 0.27
m1 = 0.106
Ctot = 6.69e-17
Cs = 0.296*Ctot
##Cd = 0.217*Ctot
Cd = Ctot/(1+m1/m2)
##Cp = 0.437*Ctot
Cp = m1*Cd

Ec = q**2/Ctot			# Charging energy (J)
EceV = Ec/q			# Charging energy (eV)

# tunnel couplings 
RL = 1.6e9			# tunnel coupling to left lead (Hz)
RR = 1.6e9          # tunnel coupling to right lead (Hz)

###########################################################################
# Sweep parameters for 1 D sweep
Vstart = 0.05;			# (V)	
Vend = 0.075;			# (V)
npts = 1000;			# number of points in sweep
delV = (Vend-Vstart)/(npts-1) 	# step size (-1 for the endpoints)

Vd = 0.0001             # Drain voltage of 0.1 mV

Vp = np.zeros(npts);		# Declare an empty 1D array for the plunger voltage that will be swept
I = np.zeros(npts);            	# Declare an empty 1D array for the current

# constant voltages
Vs = 0;				# Ground right lead 
muR = -Vs;			# Chem pot of right lead (source) in eV
Vlg = 0;			# Left barrier gate voltage (V)
Vrg = 0;			# right barrier gate voltage (V)


nstates = 6;			# Number of states considered in simulation (N= 0,1,2,3,4,5 for now)
A = np.zeros(shape=(nstates,nstates))	# Declare A matrix and fill with zeros
C = np.zeros(shape=(nstates,1))         # Declare C vector of solutions to rate equations and normalization

#Energy = np.zeros(nstates);		# Declare a 1D array for all the state energies
#Mu = np.zeros(nstates-1);		# Declare a 1D array for all the dot potentials

##########################################################################
# Define the Fermi function 
def Fermi(energy, mu,Beta):
	n = expit(-Beta*(energy-mu))
	return n;
###########################################################################
Fig1 = plt.figure(figsize=(8,8))     #create figure

###########################################################################

for k in Betas:
	for i in np.arange(npts):
		Vp[i] = Vstart + i*delV		# set plunger voltage for this point
		muL = -Vd		# declare Energies of states considered 
		E0 = (-q*(-q0 -0.5) + Cs*Vs + Cd*Vd + Cp*Vp[i])**2/(2*q*Ctot)
		E1 = (-q*(1- q0-0.5) + Cs*Vs + Cd*Vd + Cp*Vp[i])**2/(2*q*Ctot) 
		E2 = (-q*(2- q0-0.5) + Cs*Vs + Cd*Vd + Cp*Vp[i])**2/(2*q*Ctot)
		E3 = (-q*(3- q0-0.5) + Cs*Vs + Cd*Vd + Cp*Vp[i])**2/(2*q*Ctot)
		E4 = (-q*(4- q0-0.5) + Cs*Vs + Cd*Vd + Cp*Vp[i])**2/(2*q*Ctot) 
		E5 = (-q*(5- q0-0.5) + Cs*Vs + Cd*Vd + Cp*Vp[i])**2/(2*q*Ctot) 
		''' Define the elements of the matrix A of the master equations '''
		# rates between states N = 0 and N = 1
		G10 = RL*Fermi((E1-E0),muL,k) + RR*Fermi((E1-E0),muR,k)				# Rate from state N=0 to N=1
		G01 =  RL*(1-Fermi((E1-E0),muL,k)) + RR*(1 - Fermi((E1-E0),muR,k))			# Rate from state N=1 to N=0
        

		# rates between states N = 1 and N = 2
		G21 = RL*Fermi((E2-E1),muL,k) + RR*Fermi((E2-E1),muR,k)				# Rate from state N=1 to N=2
		G12 =  RL*(1-Fermi((E2-E1),muL,k)) + RR*(1 - Fermi((E2-E1),muR,k))			# Rate from state N=2 to N=1

		# rates between states N = 2 and N = 3
		G32 = RL*Fermi((E3-E2),muL,k) + RR*Fermi((E3-E2),muR,k)				# Rate from state N=3 to N=2
		G23 =  RL*(1-Fermi((E3-E2),muL,k)) + RR*(1 - Fermi((E3-E2),muR,k))			# Rate from state N=2 to N=3

		# rates between states N = 3 and N = 4
		G43 = RL*Fermi((E4-E3),muL,k) + RR*Fermi((E4-E3),muR,k)				# Rate from state N=4 to N=3
		G34 =  RL*(1-Fermi((E4-E3),muL,k)) + RR*(1 - Fermi((E4-E3),muR,k))			# Rate from state N=3 to N=4

		# rates between states N = 4 and N = 5
		G54 = RL*Fermi((E5-E4),muL,k) + RR*Fermi((E5-E4),muR,k)				# Rate from state N=5 to N=4
		G45 =  RL*(1-Fermi((E5-E4),muL,k)) + RR*(1 - Fermi((E5-E4),muR,k))			# Rate from state N=4 to N=5

		# Matrix row from the 1st master equation into and out of N=0 state
		A[0,0] = -G10
		A[0,1] = G01
		# Matrix row from the 2nd Master Equation, into and out of N=1 state
		A[1,0] = G10
		A[1,1] = -G01 - G21
		A[1,2] = G12
		# Matrix row from the 3rd Master Equation, into and out of N=2 state
		A[2,1] = G21
		A[2,2] = -G12 - G32
		A[2,3] = G23
		# Matrix row from the 4th Master Equation, into and out of N=3 state
		A[3,2] = G32
		A[3,3] = -G23 - G43 
		A[3,4] = G34
		# Matrix row from the 5th Master Equation, into and out of ed state
		A[4,3] = G43
		A[4,4] = -G34 - G54
		A[4,5] = G45
		# normalize all probabilities to 1
		A[5,0] = 1 
		A[5,1] = 1 
		A[5,2] = 1 
		A[5,3] = 1 
		A[5,4] = 1 
		A[5,5] = 1 
		# Solution vector steady state --> all 0 normalization line = 1
		C[5,0] = 1
		
		# Solve the matrix equation to get probability vector
		P = np.linalg.solve(A,C)
	
		# Calculate the source current by looking at transfer across right barrier
		on1 = P[0]*RR*Fermi((E1-E0),muR,k)
		off1 = P[1]*RR*(1-Fermi((E1-E0),muR,k))
		on2 = P[1]*RR*Fermi((E2-E1),muR,k)
		off2 = P[2]*RR*(1-Fermi((E2-E1),muR,k))
		on3 = P[2]*RR*Fermi((E3-E2),muR,k)
		off3 = P[3]*RR*(1-Fermi((E3-E2),muR,k))
		on4 = P[3]*RR*Fermi((E4-E3),muR,k)
		off4 = P[4]*RR*(1-Fermi((E4-E3),muR,k))
		on5 = P[4]*RR*Fermi((E5-E4),muR,k)
		off5 = P[5]*RR*(1-Fermi((E5-E4),muR,k))


		I[i] = q*(on1 + on2 + on3 + on4 +on5 - off1 -off2 - off3 -off4 - off5)
        
	plt.plot(Vp,I,'-',linewidth=5)


plt.xlim(Vstart,Vend)
#plt.ylim(Vdstart,Vdend)
plt.legend(LegTxt, fontsize = 20)
plt.ylabel('$I$ (A)',fontsize=20)
plt.xlabel('$V_g$ (V)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

#plt.contourf(Vp,Vd,I,100, cmap=plt.get_cmap('seismic'))
#plt.plot(Vp[:,0],I[:,0],'.-k')
#plt.contour(Vp,Vd,I,100, cmap=plt.get_cmap('seismic'))
##plt.colorbar()
#plt.show()
#
