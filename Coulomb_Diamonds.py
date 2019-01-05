''' Attempting to simulate transport through a single quantum dot
using master equation approach. This simulation includes only charge
states. 
'''

import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import numpy as np

###########################################################################
# Details of the simulation parameters
temp = 20.5;		# Temperature (of electrons in reservoirs) (K)
###########################################################################
## constants
q = 1.602e-19; 		# fundamental charge (C)
kb = 1.381e-23; 	# Boltzmann constant (J/K)
kbeV = 8.6173e-5; 	# Boltzmann constant (eV/K)
h = 4.1357e-15;		# Plank's constant (eV s)
hbar = 6.582e-16;	# Reduced Plank's constant (h/2pi) (eV s)
# Physical parameters
q0 = 0; 		# Background charge assumed to be 0
Beta = 1/(temp*kbeV)	# in inverse eV


# Device parameters, simple 3 gates - 2 barrier (left,right) and one plunger and source drain leads
# Capacitances
Cd = 1e-18;			# drain capacitance (F) 
Cs = 2.5e-18;			# source capacitance (F) 
Cp = 2e-18;			# Plunger gate capacitance (F) 
Clg = 2e-18;			# Left barrier gate capacitance (F) 
Crg = 2e-18;			# Right barrier gate capacitance (F)
Ctot = Cd + Cs + Cp + Clg + Crg;	# total dot capacitance

Ec = q**2/Ctot			# Charging energy (J)
EceV = Ec/q			# Charging energy (eV)
# tunnel couplings 
tl = 0.04e9;			# tunnel coupling to left lead (Hz)
tr = 0.04e9;			# tunnel coupling to right lead (Hz)

RL = 2*np.pi*tl**2
RR = 2*np.pi*tr**2
''' as per Beenakker PRB 44, 1646 (1991) hGamma << kbT.
Taking << to be 1%, and T = 200 mK, t is ~ 0.04 GHz '''

###########################################################################
# Sweep parameters for 1 D sweep
Vstart = -0.02;			# (V)	
Vend = 0.32;			# (V)
npts = 300;			# number of points in sweep
delV = (Vend-Vstart)/(npts-1) 	# step size (-1 for the endpoints)

Vdstart = -0.0175
Vdend = 0.025
npts2 = 200;
delVd = (Vdend-Vdstart)/(npts2-1)


Vp = np.zeros(shape=(npts,npts2));		# Declare an empty 2D array for the plunger voltage that will be swept
I = np.zeros(shape=(npts,npts2));            	# Declare an empty 2D array for the current

E0 = np.zeros(shape=(npts,npts2));		# Declare energy array for energy of states 
E1 = np.zeros(shape=(npts,npts2));
E2 = np.zeros(shape =(npts,npts2));
E3 = np.zeros(shape=(npts,npts2));
E4 = np.zeros(shape=(npts,npts2));

# Drain voltage settings
#Vd = 0.005			# Vd in V
Vd = np.zeros(shape=(npts,npts2));		# Vd in V
muL = np.zeros(shape=(npts,npts2)); 			# Chem pot of left lead (drain) in eV 

# constant voltages
Vs = 0;				# Ground right lead 
muR = -Vs;			# Chem pot of right lead (source) in eV
Vlg = 0;			# Left barrier gate voltage (V)
Vrg = 0;			# right barrier gate voltage (V)


nstates = 5;			# Number of states considered in simulation (N= 0,1,2,3,4,5 for now)
A = np.zeros(shape=(nstates,nstates))	# Declare A matrix and fill with zeros
C = np.zeros(shape=(nstates,1))         # Declare C vector of solutions to rate equations and normalization

#Energy = np.zeros(nstates);		# Declare a 1D array for all the state energies
#Mu = np.zeros(nstates-1);		# Declare a 1D array for all the dot potentials

###########################################################################
# Define the Fermi function 
def Fermi(energy, mu):
	n = 1/(1 + np.exp(Beta*(energy- mu)))
	return n;
###########################################################################

for k in np.arange(0,npts2):
	for i in np.arange(0,npts):
		Vp[i,k] = Vstart + i*delV		# set plunger voltage for this point
		Vd[i,k] = Vdstart +k*delVd
		muL[i,k] = -Vd[i,k]
		# declare Energies of states considered 
		E0[i,k] = (-q*(q0) + Cs*Vs + Cd*Vd[i,k] + Cp*Vp[i,k])**2/(2*q*Ctot)
		E1[i,k] = (-q*(1- q0) + Cs*Vs + Cd*Vd[i,k] + Cp*Vp[i,k])**2/(2*q*Ctot)
		E2[i,k] = (-q*(2- q0) + Cs*Vs + Cd*Vd[i,k] + Cp*Vp[i,k])**2/(2*q*Ctot)
		E3[i,k] = (-q*(3- q0) + Cs*Vs + Cd*Vd[i,k] + Cp*Vp[i,k])**2/(2*q*Ctot)
		E4[i,k] = (-q*(4- q0) + Cs*Vs + Cd*Vd[i,k] + Cp*Vp[i,k])**2/(2*q*Ctot)
		''' Define the elements of the matrix A of the master equations '''
		# rates in and out of state 0
		A[0,0] = RL*Fermi((E1[i,k]-E0[i,k]),muL[i,k]) + RR*Fermi((E1[i,k]-E0[i,k]),muR)				# Rate from state 0 to 1
		A[0,1] = -1.0*( RL*(1-Fermi((E1[i,k]-E0[i,k]),muL[i,k])) + RR*(1 - Fermi((E1[i,k]-E0[i,k]),muR)))	# Rate from 1 to 0
		# rates in and out of state 1
		A[1,1] = RL*Fermi((E2[i,k]-E1[i,k]),muL[i,k]) + RR*Fermi((E2[i,k]-E1[i,k]),muR)				# Rate from state 1 to 2
		A[1,2] = -1.0*( RL*(1-Fermi((E2[i,k]-E1[i,k]),muL[i,k])) + RR*(1 - Fermi((E2[i,k]-E1[i,k]),muR)))	# Rate from 2 to 1
		# rates in and out of state 2
		A[2,2] = RL*Fermi((E3[i,k]-E2[i,k]),muL[i,k]) + RR*Fermi((E3[i,k]-E2[i,k]),muR)				# Rate from state 2 to 3
		A[2,3] = -1.0*( RL*(1-Fermi((E3[i,k]-E2[i,k]),muL[i,k])) + RR*(1 - Fermi((E3[i,k]-E2[i,k]),muR)))	# Rate from 3 to 2
		# rates in and out of state 3
		A[3,3] = RL*Fermi((E4[i,k]-E3[i,k]),muL[i,k]) + RR*Fermi((E4[i,k]-E3[i,k]),muR)				# Rate from state 2 to 3
		A[3,4] = -1.0*( RL*(1-Fermi((E4[i,k]-E3[i,k]),muL[i,k])) + RR*(1 - Fermi((E4[i,k]-E3[i,k]),muR)))	# Rate from 3 to 2
		# rates in and out of state 3 give same equations as in and out of state 2 so we don't need to include them (makes matrix non-invertable)
		# normalize all probabilities to 1
		A[4,0] = 1 
		A[4,1] = 1 
		A[4,2] = 1 
		A[4,3] = 1 
		A[4,4] = 1 
		# Solution vector steady state --> all 0 normalization line = 1
		C[4,0] = 1
		#C = np.zeros(shape=(nstates,1))
		#C[nstates-1,0] = 1
		
		# Solve the matrix equation to get probability vector
		P = np.matrix.getI(A)*C
	
		# Calculate the source current by looking at transfer across right barrier
		on1 = P[0]*RR*Fermi((E1[i,k]-E0[i,k]),muR)
		off1 = P[1]*RR*(1-Fermi((E1[i,k]-E0[i,k]),muR))
		on2 = P[1]*RR*Fermi((E2[i,k]-E1[i,k]),muR)
		off2 = P[2]*RR*(1-Fermi((E2[i,k]-E1[i,k]),muR))
		on3 = P[2]*RR*Fermi((E3[i,k]-E2[i,k]),muR)
		off3 = P[3]*RR*(1-Fermi((E3[i,k]-E2[i,k]),muR))
		on4 = P[3]*RR*Fermi((E4[i,k]-E3[i,k]),muR)
		off4 = P[4]*RR*(1-Fermi((E4[i,k]-E3[i,k]),muR))
	
		I[i,k] = -q*(on1 + on2 + on3 + on4 - off1 -off2 - off3 -off4)
	

#plt.plot(Vp,E0,'-k')
#plt.plot(Vp,E1,'-r')
#plt.plot(Vp,E2,'-b')
#plt.plot(Vp,E3,'-g')
#plt.show()
#plt.plot(Vp,I,'.-k')
plt.pcolormesh(Vp,Vd,I, cmap=plt.get_cmap('seismic'))
plt.show()
