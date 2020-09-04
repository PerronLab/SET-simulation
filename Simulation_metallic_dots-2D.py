''' 
Simulation of transport through a quantum dot using a master equation approach
Supplemental information from the paper
"Transport Through Quantum Dots: An Introduction via Master Equation Simulations"
Published 2020. 
authors: Robert A. Bush, Erick D. Ochoa, and Justin K. Perron
'''
#import matplotlib
import matplotlib.pyplot as plt
#from matplotlib.mlab import griddata
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


# Device parameters, 
# Capacitances
Cd = 4.80e-17
Cs = 1.98e-17
Cp = 5.09e-18
Ctot = Cd+Cs+Cp

Ec = q**2/Ctot			# Charging energy (J)
EceV = Ec/q			# Charging energy (eV)

# tunnel rates

RL = 1.6e6 	#left barrier	
RR = 1.6e6	#right barrier 

###########################################################################
# Sweep parameters for 1 D sweep
Vstart = -0.03;		# Starting gate voltage (V)	
Vend = 0.1;			# final gate voltage (V)
npts = 500;			# number of points in Vg sweep
delV = (Vend-Vstart)/(npts-1) 	# step size (-1 for the endpoints)

Vdstart = -0.0025		# starting Drain voltage
Vdend = 0.0025			# final Drain voltage
npts2 = 200;			# number of points for Vd measurement
delVd = (Vdend-Vdstart)/(npts2-1) # step size


Vp = np.zeros(shape=(npts,npts2));		# Declare an empty 2D array for the plunger voltage that will be swept
I = np.zeros(shape=(npts,npts2));            	# Declare an empty 2D array for the current

# Drain voltage settings
Vd = np.zeros(shape=(npts,npts2));		# Declare an 2D array for drain voltage

Vs = 0;				# Ground right lead 
muR = -Vs;			# Chem pot of right lead (source) in eV

nstates = 6;			# Number of states considered in simulation (N= 0,1,2,3,4,5 for now)
A = np.zeros(shape=(nstates,nstates))	# Declare A matrix and fill with zeros
C = np.zeros(shape=(nstates,1))         # Declare C vector of solutions to rate equations and normalization

# Define the Fermi function 
def Fermi(energy, mu):
	n = expit(-Beta*(energy-mu))
	return n;

for k in np.arange(0,npts2):
	for i in np.arange(0,npts):
		Vp[i,k] = Vstart + i*delV		# set gate voltage 
		Vd[i,k] = Vdstart +k*delVd		# set the drain voltage  
		muL = -Vd[i,k]
		# declare Energies of states considered 
		E0 = (-q*(-q0 -0.5) + Cs*Vs + Cd*Vd[i,k] + Cp*Vp[i,k])**2/(2*q*Ctot)
		E1 = (-q*(1- q0-0.5) + Cs*Vs + Cd*Vd[i,k] + Cp*Vp[i,k])**2/(2*q*Ctot) 
		E2 = (-q*(2- q0-0.5) + Cs*Vs + Cd*Vd[i,k] + Cp*Vp[i,k])**2/(2*q*Ctot)
		E3 = (-q*(3- q0-0.5) + Cs*Vs + Cd*Vd[i,k] + Cp*Vp[i,k])**2/(2*q*Ctot)
		E4 = (-q*(4- q0-0.5) + Cs*Vs + Cd*Vd[i,k] + Cp*Vp[i,k])**2/(2*q*Ctot) 
		E5 = (-q*(5- q0-0.5) + Cs*Vs + Cd*Vd[i,k] + Cp*Vp[i,k])**2/(2*q*Ctot) 
		# rates between states N = 0 and N = 1
		G10 = RL*Fermi((E1-E0),muL) + RR*Fermi((E1-E0),muR)				# Rate from state N=0 to N=1
		G01 =  RL*(1-Fermi((E1-E0),muL)) + RR*(1 - Fermi((E1-E0),muR))			# Rate from state N=1 to N=0
        

		# rates between states N = 1 and N = 2
		G21 = RL*Fermi((E2-E1),muL) + RR*Fermi((E2-E1),muR)				# Rate from state N=1 to N=2
		G12 =  RL*(1-Fermi((E2-E1),muL)) + RR*(1 - Fermi((E2-E1),muR))			# Rate from state N=2 to N=1

		# rates between states N = 2 and N = 3
		G32 = RL*Fermi((E3-E2),muL) + RR*Fermi((E3-E2),muR)				# Rate from state N=3 to N=2
		G23 =  RL*(1-Fermi((E3-E2),muL)) + RR*(1 - Fermi((E3-E2),muR))			# Rate from state N=2 to N=3

		# rates between states N = 3 and N = 4
		G43 = RL*Fermi((E4-E3),muL) + RR*Fermi((E4-E3),muR)				# Rate from state N=4 to N=3
		G34 =  RL*(1-Fermi((E4-E3),muL)) + RR*(1 - Fermi((E4-E3),muR))			# Rate from state N=3 to N=4

		# rates between states N = 4 and N = 5
		G54 = RL*Fermi((E5-E4),muL) + RR*Fermi((E5-E4),muR)				# Rate from state N=5 to N=4
		G45 =  RL*(1-Fermi((E5-E4),muL)) + RR*(1 - Fermi((E5-E4),muR))			# Rate from state N=4 to N=5

		# Define the elements of the matrix A of the master equations 
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
		on1 = P[0]*RR*Fermi((E1-E0),muR)
		off1 = P[1]*RR*(1-Fermi((E1-E0),muR))
		on2 = P[1]*RR*Fermi((E2-E1),muR)
		off2 = P[2]*RR*(1-Fermi((E2-E1),muR))
		on3 = P[2]*RR*Fermi((E3-E2),muR)
		off3 = P[3]*RR*(1-Fermi((E3-E2),muR))
		on4 = P[3]*RR*Fermi((E4-E3),muR)
		off4 = P[4]*RR*(1-Fermi((E4-E3),muR))
		on5 = P[4]*RR*Fermi((E5-E4),muR)
		off5 = P[5]*RR*(1-Fermi((E5-E4),muR))


		I[i,k] = q*(on1 + on2 + on3 + on4 +on5 - off1 -off2 - off3 -off4 - off5)
	

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.pcolormesh(Vp,Vd*100,I*1e12, cmap=plt.get_cmap('seismic'))
clb = plt.colorbar()
clb.ax.set_title('$I (pA)$',fontsize = 60)
clb.ax.tick_params(labelsize=50)

#plt.xlim(Vstart,Vend)
plt.xlim(-0.02,0.1)
plt.ylim(Vdstart*100,Vdend*100)
plt.ylabel('$V_d$ (mV)',fontsize=70)
plt.xlabel('$V_g$ (V)',fontsize=70)
plt.xticks(fontsize=70)
plt.yticks(fontsize=70)


plt.show()

