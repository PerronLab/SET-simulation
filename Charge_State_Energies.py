''' 
Simulation of transport through a quantum dot using a master equation approach
Supplemental information from the paper
"Transport Through Quantum Dots: An Introduction via Master Equation Simulations"
Published 2020. 
authors: Robert A. Bush, Erick D. Ochoa, and Justin K. Perron
'''

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


# Device parameters, 
# capacitances from slopes of diamonds

Cd = 4.80e-17
Cs = 1.98e-17
Cp = 5.09e-18
Ctot = Cd+Cs+Cp

Ec = q**2/Ctot			# Charging energy (J)
EceV = Ec/q			# Charging energy (eV)

# tunnel rates 
RL = 1.6e6
RR = 1.6e6

###########################################################################
# Sweep parameters for 1 D sweep
Vstart = -0.2;			# (V)	
Vend = 0.4  ;			# (V)
npts = 200;			# number of points in sweep
delV = (Vend-Vstart)/(npts-1) 	# step size (-1 for the endpoints)


Vp = np.zeros(npts);		# Declare an empty 1D array for the plunger voltage that will be swept
I = np.zeros(npts);            	# Declare an empty 1D array for the current

E0 = np.zeros(npts);		# Declare energy array for energy of states 
E1 = np.zeros(npts);
E2 = np.zeros(npts);
E3 = np.zeros(npts);
E4 = np.zeros(npts);
E5 = np.zeros(npts);

# constant voltages
Vs = 0;				# Ground source lead 
muR = -Vs;			# Chem pot of right lead (source) in eV
Vd = 0.0          # Drain bias of 1 mV
Vlg = 0;			# Left barrier gate voltage (V)
Vrg = 0;			# right barrier gate voltage (V)

###########################################################################
# Define the Fermi function 
def Fermi(energy, mu):
	#n = 1/(1 + (Beta*(energy- mu)))
	n = expit(-Beta*(energy-mu))
	return n;
###########################################################################

for i in np.arange(0,npts):
	Vp[i] = Vstart + i*delV		# set plunger voltage for this point
	muL = -Vd
	# declare Energies of states considered 
	E0[i] = (-q*(-q0 -0.5) + Cs*Vs + Cd*Vd + Cp*Vp[i])**2/(2*q*Ctot)
	E1[i] = (-q*(1- q0-0.5) + Cs*Vs + Cd*Vd + Cp*Vp[i])**2/(2*q*Ctot) 
	E2[i] = (-q*(2- q0-0.5) + Cs*Vs + Cd*Vd + Cp*Vp[i])**2/(2*q*Ctot)
	E3[i] = (-q*(3- q0-0.5) + Cs*Vs + Cd*Vd + Cp*Vp[i])**2/(2*q*Ctot)
	E4[i] = (-q*(4- q0-0.5) + Cs*Vs + Cd*Vd + Cp*Vp[i])**2/(2*q*Ctot) 
	E5[i] = (-q*(5- q0-0.5) + Cs*Vs + Cd*Vd + Cp*Vp[i])**2/(2*q*Ctot) 

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

LegendTxt = ('N=0 state', 'N=1 state', 'N=2 state', 'N=3 state')#, 'N=4 state', 'N=5 state')
plt.plot(Vp,E0*100,'-k',linewidth=3)
plt.plot(Vp,E1*100,'-b',linewidth=3)
plt.plot(Vp,E2*100,'-r',linewidth=3)
plt.plot(Vp,E3*100,'-g',linewidth=3)
#plt.plot(Vp,E4,'-y',linewidth=3)
#plt.plot(Vp,E5,'-c',linewidth=3)
plt.xlim(-0.1,0.18)
plt.ylim(-0.1,4)
plt.ylabel('$U(N)$ (meV)',fontsize=70)
plt.xlabel('$V_g$ (V)',fontsize=70)
plt.xticks(fontsize=70)
plt.yticks(fontsize=70)
#leg = plt.legend(LegendTxt, loc='upper center', fontsize='xx-large')
leg = plt.legend(LegendTxt, loc='upper center', prop={'size':70     })
for legobj in leg.legendHandles:
    legobj.set_linewidth(15.0)
plt.show()

