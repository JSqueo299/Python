import cantera as ct
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Generating adiabatic flame temperature for N mechanism files

# Setting Tad to empty array
S_L = []

# Number of mechanisms being considered
N = 30
n = np.arange(1,N+1)


#'''
## Setting Simulation parameters
p = ct.one_atm  # pressure [Pa]
#phi = float(sys.argv[1])
phi = 0.7
T = 300.0  # unburned gas temperature [K]
reactants = 'CH4:1, O2:2, N2:7.52'  # premixed gas composition
width = 0.03  # m
loglevel = 0  # amount of diagnostic output (0 to 8)

for i in n:
	i = str(i)
	#mechanism = '25_perturbed_ch4_mech_' + i + '.cti'
	#gas = ct.Solution(mechanism)
    gas = Solution(’gri30.cti’)

	#gas.set_equivalence_ratio(phi, 'CH4', {'O2':2.0, 'N2':7.52})
	#gas.TP = Tin, p
	
	gas.TPX=T,p,{'CH4':phi,'O2':2,'N2':2*3.76}

	# Set up flame object
	f = ct.FreeFlame(gas, width=width)
	f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)

	# Solve with mixture-averaged transport model
	f.transport_model = 'Mix'
	f.solve(loglevel=loglevel, auto=True)


	# Calculating response functions
	dTdx = np.gradient(f.T)
	first_half_max = np.argmin(np.abs(dTdx-np.amax(dTdx)/2.0))
	second_half_max = np.argmin(np.abs(dTdx[first_half_max+2:]-np.amax(dTdx/2.0)))+(first_half_max+2)
	full_width = f.grid[second_half_max] - f.grid[first_half_max]
	final_x = f.grid[-1]

	S_L_ind = f.u[0]
	S_L.append(S_L_ind)
	
	with open('SL.csv', 'a') as fid:
	    fid.write(str(f.u[0])+'\n')
	
	#f.T[-1].append(Tad)
	#print('equivalence ratio = {0:7f}'.format(phi))
	#print(mechanism)
	#print('flamespeed for = {0:7f} m/s'.format(f.u[0]))
	#print('--------------------------------------------')
	#print('temperature = {0:7f} K'.format(f.T[-1]))
	#print('thickness = {0:7f} mm'.format(full_width*1e3))


