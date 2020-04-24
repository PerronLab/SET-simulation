''' 
Simulation of transport through a quantum dot using a master equation approach
Supplemental information from the paper
"Transport Through Quantum Dots: An Introduction via Master Equation Simulations"
Published 2020. 
authors: Robert A. Bush, Erick D. Ochoa, and Justin K. Perron
'''
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import expit
import numpy as np

###########################################################################
# Details of the simulation parameters
temp = 0.25;		# Temperature (of electrons in reservoirs) (K)
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


# Device parameters, Estimated from Hanson et al. (2007) Rev Mod Phys 79, 4 1217-1265

m2 = 0.27	
m1 = 0.106	
Ctot = 6.69e-17	
Cs = 0.296*Ctot
Cd = Ctot/(1+m1/m2)
Cp = m1*Cd

#Calculation of g factor as a function of magnetic field as defined by Hansen et. al. (2003)
def g(B):
        #return 0.29     		# "Forced to be linear we get |g| 0.29" - just below eqn 1 in PRL
        return gCon - 0.0077*B		# Equation 1 in PRL

#Quantum energy contributions.
B = 6		# Applied magnetic field (T)	
orbital = 1.22e-3
gCon = 0.43
Magneton = 9.274e-24
zeeman = g(B)*Magneton*B/q

Gorb = 3e5;         # 0.30 MHz from PRL 91, 196802 (2003) page 3

Ec = q**2/Ctot			# Charging energy (J)
EceV = Ec/q			# Charging energy (eV)

# tunnel couplings 
RL = 1.7e6          # 1.7 MHz from PRL 91, 196802 (2003) page 3
RR = 1.7e6          # 1.7 MHz from PRL 91, 196802 (2003) page 3 (we assume the rates of both barriers are equal during the Coulomb diamond measurement)

###########################################################################
# Sweep parameters for 1 D sweep
Vstart = -0.008;			# (V)	
Vend = 0.01;			# (V)
npts = 200;			# number of points in sweep
delV = (Vend-Vstart)/(npts-1) 	# step size (-1 for the endpoints)

Vdstart = 0
Vdend = 0.002
npts2 = 200;
delVd = (Vdend-Vdstart)/(npts2-1)


Vp = np.zeros(shape=(npts,npts2));		# Declare an empty 2D array for the plunger voltage that will be swept
I = np.zeros(shape=(npts,npts2));            	# Declare an empty 2D array for the current

# Drain voltage settings
Vd = np.zeros(shape=(npts,npts2));		# Vd in V

# constant voltages
Vs = 0;				# Ground right lead 
muR = -Vs;			# Chem pot of right lead (source) in eV

nstates = 6;			# Number of states considered in simulation (N= 0,1gu, 1gd, 1eu, 1ed, 2)
A = np.zeros(shape=(nstates,nstates))	# Declare A matrix and fill with zeros
C = np.zeros(shape=(nstates,1))         # Declare C vector of solutions to rate equations and normalization

###########################################################################
# Define the Fermi function 
def Fermi(energy, mu):
	n = expit(-Beta*(energy-mu))
	return n;
###########################################################################

for k in np.arange(0,npts2):
	for i in np.arange(0,npts):
		Vp[i,k] = Vstart + i*delV		# set plunger voltage for this point
		Vd[i,k] = Vdstart +k*delVd
		muL = -Vd[i,k]
		# declare Energies of states considered 
		E0 = (-q*(-q0 -0.5) + Cs*Vs + Cd*Vd[i,k] + Cp*Vp[i,k])**2/(2*q*Ctot)
		E1 = (-q*(1- q0-0.5) + Cs*Vs + Cd*Vd[i,k] + Cp*Vp[i,k])**2/(2*q*Ctot) - zeeman
		E2 = (-q*(1- q0-0.5) + Cs*Vs + Cd*Vd[i,k] + Cp*Vp[i,k])**2/(2*q*Ctot) + zeeman
		E3 = (-q*(1- q0-0.5) + Cs*Vs + Cd*Vd[i,k] + Cp*Vp[i,k])**2/(2*q*Ctot) - zeeman + orbital
		E4 = (-q*(1- q0-0.5) + Cs*Vs + Cd*Vd[i,k] + Cp*Vp[i,k])**2/(2*q*Ctot) + zeeman + orbital
		E5 = (-q*(2- q0-0.5) + Cs*Vs + Cd*Vd[i,k] + Cp*Vp[i,k])**2/(2*q*Ctot) + 2*orbital
		''' Define the elements of the matrix A of the master equations '''
		# rates in and out of state 0
		G10 = RL*Fermi((E1-E0),muL) + RR*Fermi((E1-E0),muR)				# Rate from state 0 to gu
		G20 = RL*Fermi((E2-E0),muL) + RR*Fermi((E2-E0),muR)				# Rate from state 0 to gd
		G30 = RL*Fermi((E3-E0),muL) + RR*Fermi((E3-E0),muR)				# Rate from state 0 to eu
		G40 = RL*Fermi((E4-E0),muL) + RR*Fermi((E4-E0),muR)				# Rate from state 0 to ed

		G01 =  RL*(1-Fermi((E1-E0),muL)) + RR*(1 - Fermi((E1-E0),muR))			# Rate from gu to 0
		G02 =  RL*(1-Fermi((E2-E0),muL)) + RR*(1 - Fermi((E2-E0),muR))			# Rate from gd to 0
		G03 =  RL*(1-Fermi((E3-E0),muL)) + RR*(1 - Fermi((E3-E0),muR))			# Rate from eu to 0
		G04 =  RL*(1-Fermi((E4-E0),muL)) + RR*(1 - Fermi((E4-E0),muR))			# Rate from ed to 0
        
		G13 = Gorb	# relation rate from excited orbital to ground
		G24 = Gorb	# relation rate from excited orbital to ground

		G51 = RL*Fermi((E5-E1),muL) + RR*Fermi((E5-E1),muR)				# Rate from state gu to N=2
		G52 = RL*Fermi((E5-E2),muL) + RR*Fermi((E5-E2),muR)				# Rate from state gd to N=2
		G53 = RL*Fermi((E5-E3),muL) + RR*Fermi((E5-E3),muR)				# Rate from state eu to N=2
		G54 = RL*Fermi((E5-E4),muL) + RR*Fermi((E5-E4),muR)				# Rate from state ed to N=2

		G15 =  RL*(1-Fermi((E5-E1),muL)) + RR*(1 - Fermi((E5-E1),muR))			# Rate from gu to 0
		G25 =  RL*(1-Fermi((E5-E2),muL)) + RR*(1 - Fermi((E5-E2),muR))			# Rate from gd to 0
		G35 =  RL*(1-Fermi((E5-E3),muL)) + RR*(1 - Fermi((E5-E3),muR))			# Rate from eu to 0
		G45 =  RL*(1-Fermi((E5-E4),muL)) + RR*(1 - Fermi((E5-E4),muR))			# Rate from ed to 0
		# Matrix row from the 1st master equation into and out of N=0 state
		A[0,0] = -(G10+G20+G30+G40) 										# Matrix element 00
		A[0,1] = G01											 	# Matrix element 01 
		A[0,2] = G02												# Matrix element 02
		A[0,3] = G03												# Matrix element 03
		A[0,4] = G04												# Matrix element 04
		# Matrix row from the 2nd Master Equation, into and out of gu state
		A[1,0] = G10
		A[1,1] = -G01 - G51
		A[1,3] = G13
		A[1,5] = G15
		# Matrix row from the 3rd Master Equation, into and out of gd state
		A[2,0] = G20
		A[2,2] = -G02 - G52
		A[2,4] = G24
		A[2,5] = G25
		# Matrix row from the 4th Master Equation, into and out of eu state
		A[3,0] = G30
		A[3,3] = -G03 - G53 - G13
		A[3,5] = G35
		# Matrix row from the 5th Master Equation, into and out of ed state
		A[4,0] = G40
		A[4,4] = -G04 - G54 -G24
		A[4,5] = G45
		# normalize the sum of all probabilities to 1
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

		on1 = P[0]*RR*Fermi((E1-E0),muR)
		off1 = P[1]*RR*(1-Fermi((E1-E0),muR))
		on2 = P[0]*RR*Fermi((E2-E0),muR)
		off2 = P[2]*RR*(1-Fermi((E2-E0),muR))
		on3 = P[0]*RR*Fermi((E3-E0),muR)
		off3 = P[3]*RR*(1-Fermi((E3-E0),muR))
		on4 = P[0]*RR*Fermi((E4-E0),muR)
		off4 = P[4]*RR*(1-Fermi((E4-E0),muR))

		on5a = P[1]*RR*Fermi((E5-E1),muR)
		on5b = P[2]*RR*Fermi((E5-E2),muR)
		on5c = P[3]*RR*Fermi((E5-E3),muR)
		on5d = P[4]*RR*Fermi((E5-E4),muR)
		on5 = on5a+on5b+on5c+on5d

		# Not sure if we should allow transitions from ground N=2 state to an excited orbital state of N=1. Thinking not, but it's working soooooo
		off5a = P[5]*RR*(1-Fermi((E5-E1),muR))
		off5b = P[5]*RR*(1-Fermi((E5-E2),muR))
		off5c = P[5]*RR*(1-Fermi((E5-E3),muR))
		off5d = P[5]*RR*(1-Fermi((E5-E4),muR))
		off5 = off5a+off5b+off5c+off5d

		I[i,k] = -q*(on1 + on2 + on3 + on4 +on5 - off1 -off2 - off3 -off4 - off5)
	

gradI = np.gradient(I)	#take the gradient of the DC current to get AC data

# Plot the simulation
plt.rc('xtick',labelsize=15)	#
plt.rc('ytick',labelsize=15)
plt.pcolormesh((Vp-Vp[0])*10**3, Vd*10**3, gradI[0], cmap=plt.get_cmap('gray'))
plt.ylabel('$V_d$ (mV)',fontsize=20)
plt.xlabel('$\Delta V_g$ (mV)',fontsize=20)
plt.show()
