''' Attempting to simulate transport through a single quantum dot
using master equation approach. This simulation includes only charge
states. 
'''

import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from scipy.special import expit
import numpy as np

###########################################################################
# Details of the simulation parameters
temp = 1.5;		# Temperature (of electrons in reservoirs) (K)
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
Ctot = 6.68e-17
Cs = 0.296*Ctot
Cd = 0.217*Ctot
Cp = 0.487*Ctot

#Quantum energy contributions.
B = 14
orbital = 1.1e-3
g = 0.43
Magneton = 9.274e-24
zeeman = g*Magneton*B/q

Ec = q**2/Ctot			# Charging energy (J)
EceV = Ec/q			# Charging energy (eV)

# tunnel couplings 
RL = 100e6
RR = 1.6e6

###########################################################################
# Sweep parameters for 1 D sweep
Vstart = -0.01;			# (V)	
Vend = 0.05;			# (V)
npts = 400;			# number of points in sweep
delV = (Vend-Vstart)/(npts-1) 	# step size (-1 for the endpoints)

#Vdstart = -0.02
#Vdend = 0.0015
#npts2 = 3;
#delVd = (Vdend-Vdstart)/(npts2-1)


#Vp = np.zeros(shape=(npts,npts2));		# Declare an empty 2D array for the plunger voltage that will be swept
Vp = np.zeros(npts)
#I = np.zeros(shape=(npts,npts2));            	# Declare an empty 2D array for the current
I = np.zeros(npts)
#
#E0 = np.zeros(shape=(npts,npts2));		# Declare energy array for energy of states 
E0 = np.zeros(npts)
E1 = np.zeros(npts)
E2 = np.zeros(npts)
E3 = np.zeros(npts)
E4 = np.zeros(npts)
E5 = np.zeros(npts)

mu0 = np.zeros(npts)
mu1 = np.zeros(npts)
mu2 = np.zeros(npts)
mu3 = np.zeros(npts)
mu4 = np.zeros(npts)
mu5 = np.zeros(npts)
#E1 = np.zeros(shape=(npts,npts2));
#E2 = np.zeros(shape =(npts,npts2));
#E3 = np.zeros(shape=(npts,npts2));
#E4 = np.zeros(shape=(npts,npts2));
#E5 = np.zeros(shape=(npts,npts2));

PA = np.zeros(npts)
PB = np.zeros(npts)
PC = np.zeros(npts)
PD = np.zeros(npts)
PE = np.zeros(npts)
PF = np.zeros(npts)
# Drain voltage settings
Vd = -0.0051			# Vd in V
#Vd = np.zeros(shape=(npts,npts2));		# Vd in V
#muL = np.zeros(shape=(npts,npts2)); 			# Chem pot of left lead (drain) in eV 

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

###########################################################################
# Define the Fermi function 
def Fermi(energy, mu):
	#n = 1/(1 + (Beta*(energy- mu)))
	n = expit(-Beta*(energy-mu))
	return n;
###########################################################################

muL = -Vd
for i in np.arange(0,npts):
	Vp[i] = Vstart + i*delV		# set plunger voltage for this point
	# declare Energies of states considered 
	E0[i] = (-q*(q0) + Cs*Vs + Cd*Vd + Cp*Vp[i])**2/(2*q*Ctot)
	E1[i] = (-q*(1- q0) + Cs*Vs + Cd*Vd + Cp*Vp[i])**2/(2*q*Ctot) - zeeman
	E2[i] = (-q*(1- q0) + Cs*Vs + Cd*Vd + Cp*Vp[i])**2/(2*q*Ctot) + zeeman
	E3[i] = (-q*(1- q0) + Cs*Vs + Cd*Vd + Cp*Vp[i])**2/(2*q*Ctot) - zeeman + orbital
	E4[i] = (-q*(1- q0) + Cs*Vs + Cd*Vd + Cp*Vp[i])**2/(2*q*Ctot) + zeeman + orbital
	E5[i] = (-q*(2- q0) + Cs*Vs + Cd*Vd + Cp*Vp[i])**2/(2*q*Ctot) + 2*orbital

	mu1[i] = E1[i]-E0[i]
	mu2[i] = E2[i]-E0[i]
	mu3[i] = E3[i]-E0[i]
	mu4[i] = E4[i]-E0[i]
	mu5[i] = E5[i]-E1[i]
	''' Define the elements of the matrix A of the master equations '''
	# rates in and out of state 0
	G10 = RL*Fermi((E1[i]-E0[i]),muL) + RR*Fermi((E1[i]-E0[i]),muR)				# Rate from state 0 to gu
	G20 = RL*Fermi((E2[i]-E0[i]),muL) + RR*Fermi((E2[i]-E0[i]),muR)				# Rate from state 0 to gd
	G30 = RL*Fermi((E3[i]-E0[i]),muL) + RR*Fermi((E3[i]-E0[i]),muR)				# Rate from state 0 to eu
	G40 = RL*Fermi((E4[i]-E0[i]),muL) + RR*Fermi((E4[i]-E0[i]),muR)				# Rate from state 0 to ed

	G01 =  RL*(1-Fermi((E1[i]-E0[i]),muL)) + RR*(1 - Fermi((E1[i]-E0[i]),muR))			# Rate from gu to 0
	G02 =  RL*(1-Fermi((E2[i]-E0[i]),muL)) + RR*(1 - Fermi((E2[i]-E0[i]),muR))			# Rate from gd to 0
	G03 =  RL*(1-Fermi((E3[i]-E0[i]),muL)) + RR*(1 - Fermi((E3[i]-E0[i]),muR))			# Rate from eu to 0
	G04 =  RL*(1-Fermi((E4[i]-E0[i]),muL)) + RR*(1 - Fermi((E4[i]-E0[i]),muR))			# Rate from ed to 0

	G51 = RL*Fermi((E5[i]-E1[i]),muL) + RR*Fermi((E5[i]-E1[i]),muR)				# Rate from state gu to N=2
	G52 = RL*Fermi((E5[i]-E2[i]),muL) + RR*Fermi((E5[i]-E2[i]),muR)				# Rate from state gd to N=2
	G53 = RL*Fermi((E5[i]-E3[i]),muL) + RR*Fermi((E5[i]-E3[i]),muR)				# Rate from state eu to N=2
	G54 = RL*Fermi((E5[i]-E4[i]),muL) + RR*Fermi((E5[i]-E4[i]),muR)				# Rate from state ed to N=2

	G15 =  RL*(1-Fermi((E5[i]-E1[i]),muL)) + RR*(1 - Fermi((E5[i]-E1[i]),muR))			# Rate from gu to 0
	G25 =  RL*(1-Fermi((E5[i]-E2[i]),muL)) + RR*(1 - Fermi((E5[i]-E2[i]),muR))			# Rate from gd to 0
	G35 =  RL*(1-Fermi((E5[i]-E3[i]),muL)) + RR*(1 - Fermi((E5[i]-E3[i]),muR))			# Rate from eu to 0
	G45 =  RL*(1-Fermi((E5[i]-E4[i]),muL)) + RR*(1 - Fermi((E5[i]-E4[i]),muR))			# Rate from ed to 0


	# Matrix row from the 1st master equation into and out of N=0 state
	A[0,0] = -(G10+G20+G30+G40) 										# Matrix element 00
	A[0,1] = G01											 	# Matrix element 01 
	A[0,2] = G02												# Matrix element 02
	A[0,3] = G03												# Matrix element 03
	A[0,4] = G04												# Matrix element 04
	# Matrix row from the 2nd Master Equation, into and out of gu state
	A[1,0] = G10
	A[1,1] = -G01 - G51
	A[1,5] = G15
	# Matrix row from the 3rd Master Equation, into and out of gu state
	A[2,0] = G20
	A[2,2] = -G02 - G52
	A[2,5] = G25
	# Matrix row from the 4th Master Equation, into and out of gu state
	A[3,0] = G30
	A[3,3] = -G03 - G53
	A[3,5] = G35
	# Matrix row from the 5th Master Equation, into and out of gu state
	A[4,0] = G40
	A[4,4] = -G04 - G54
	A[4,5] = G45
	# rates in and out of state 3 give same equations as in and out of state 2 so we don't need to include them (makes matrix non-invertable)
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
	P = np.matrix.getI(A)*C
	
	PA[i] = P[0]
	PB[i] = P[1]
	PC[i] = P[2]
	PD[i] = P[3]
	PE[i] = P[4]
	PF[i] = P[5]

	# Calculate the source current by looking at transfer across right barrier

	on1 = P[0]*RR*Fermi((E1[i]-E0[i]),muR)
	on2 = P[0]*RR*Fermi((E2[i]-E0[i]),muR)
	on3 = P[0]*RR*Fermi((E3[i]-E0[i]),muR)
	on4 = P[0]*RR*Fermi((E4[i]-E0[i]),muR)

	off1 = P[1]*RR*(1-Fermi((E1[i]-E0[i]),muR))
	off2 = P[2]*RR*(1-Fermi((E2[i]-E0[i]),muR))
	off3 = P[3]*RR*(1-Fermi((E3[i]-E0[i]),muR))
	off4 = P[4]*RR*(1-Fermi((E4[i]-E0[i]),muR))

	on5a = P[1]*RR*Fermi((E5[i]-E1[i]),muR)
	on5b = P[2]*RR*Fermi((E5[i]-E2[i]),muR)
	on5c = P[3]*RR*Fermi((E5[i]-E3[i]),muR)
	on5d = P[4]*RR*Fermi((E5[i]-E4[i]),muR)
	on5 = on5a+on5b+on5c+on5d

	off5a = P[5]*RR*(1-Fermi((E5[i]-E1[i]),muR))
	off5b = P[5]*RR*(1-Fermi((E5[i]-E2[i]),muR))
	off5c = P[5]*RR*(1-Fermi((E5[i]-E3[i]),muR))
	off5d = P[5]*RR*(1-Fermi((E5[i]-E4[i]),muR))
	off5 = off5a+off5b+off5c+off5d

	I[i] = -q*(on1 + on2 + on3 + on4 +on5 - off1 -off2 - off3 -off4 - off5)


#plt.plot(Vp,E0,'-k')
#plt.plot(Vp,E1,'-r')
#plt.plot(Vp,E2,'-b')
#plt.plot(Vp,E3,'-g')
#plt.show()
plt.plot(Vp,I,'.-k')
#plt.pcolormesh(Vp,Vd,I, cmap=plt.get_cmap('seismic'))
#plt.contourf(Vp,Vd,I,100, cmap=plt.get_cmap('seismic'))
#plt.plot(Vp,E0,'.-k')
#plt.plot(Vp,E1,'.-b')
#plt.plot(Vp,E2,'.-g')
#plt.plot(Vp,E3,'.-r')
#plt.plot(Vp,E4,'.-y')
#plt.plot(Vp,E5,'.-k')
#plt.contour(Vp,Vd,I,100, cmap=plt.get_cmap('seismic'))
#plt.colorbar()
plt.ion()
plt.show()
