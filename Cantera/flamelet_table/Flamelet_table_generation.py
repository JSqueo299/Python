# ===========================  FLAMELET TABLE GENERATION  =========================== #
# SOURCES:
# 1. https://www.et.byu.edu/~tom/classes/641/Cantera/Workshop/Flamelet-Cantera-July-2004.pdf
# 2. https://groups.google.com/forum/#!topic/cantera-users/VenkSlGjnb4
# 3. https://www.cerfacs.fr/cantera/docs/tutorials/CANTERA_HandsOn.pdf

# DESCRIPTION:
# 1D counterflow diffusion flame problem is used to construct a flamelet library that is thenloaded in a LES solver. This Python script iterates initializes a 1D counterflow diffusion flame then solves the flame solution iteratively. The script starts with a small jet velocity then increases strain by increasing the flow rates fo the fuel and oxidizer jets for some number of flamelets specified by 'Nfl'. A flame extinction criteria decides the flame is extinguished when the maximum temeprature of the flame is less than 900K. Flame solution is written to .xml and .csv files, which tabulate mixture fraction, axial position, flame speed, strain, temperature andd species concentrations in either mass or mole fractions for each strain. 
# =================================================================================== #

from __future__ import print_function
from __future__ import division

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

from cantera import *
from numpy import *
from types import *

print("Running Cantera Version: " + str(ct.__version__))
print("\nAn opposed-flow methane/air diffusion flame")





# Initialization of Reference conditions
p = ct.one_atm  # pressure
tin_f = 300.0   # fuel inlet temperature
tin_o = 300.0   # oxidizer inlet temperature
mdot_o = 0.5    # kg/m^2/s
mdot_f = 0.25   # kg/m^2/s
L = 0.02        # width between burners (m)
comp_o = 'O2:0.21, N2:0.78, AR:0.01'; # air composition
comp_f = 'CH4:1';                     # fuel composition

# distance between inlets is 2 cm; start with an evenly-spaced 11-point grid
N = 11   # number of inital grid points
initial_grid = linspace(0, L, N)  # generate initial grid
Nfl=15  # number of flamalets to compute
dx = L/N # grid spacing 





# Here we use GRI-Mech 3.0 with mixture-averaged transport properties.
#gas = ct.Solution('GRI_mech_cti.cti')
reaction_mechanism="gri30.xml"
gas = ct.Solution('gri30.xml', 'gri30_mix')
# Number of species
nsp = gas.n_species
# Number of elements
nel = gas.n_elements
# Molar Masses
mmas = gas.molecular_weights
# Atomic Weights
awgt = gas.atomic_weights

Fuel = gas.species_index('CH4')



# Create flame
f = CounterflowDiffusionFlame(gas, initial_grid)

#Set error tolerances
tol_ss = [1.0e-5, 1.0e-11] # [rtol, atol] for steady-state problem
tol_ts = [1.0e-5, 1.0e-11] # [rtol, atol] for time stepping
f.flame.set_steady_tolerances(default=tol_ss)
f.flame.set_transient_tolerances(default=tol_ts)

f.fuel_inlet.mdot = mdot_f
f.fuel_inlet.X = comp_f
f.fuel_inlet.T = tin_f

f.oxidizer_inlet.mdot = mdot_o
f.oxidizer_inlet.X = comp_o
f.oxidizer_inlet.T = tin_o

f.set_initial_guess()

##########################################################################
# Define a limit for the maximum temperature below which the flame is
# considered as extinguished and the computation is aborted
# This increases the speed of refinement is enabled
temperature_limit_extinction = 900  # K

def interrupt_extinction(t):
    if np.max(f.T) < temperature_limit_extinction:
        raise Exception('Flame extinguished')
    return 0.
f.set_interrupt(interrupt_extinction)
##########################################################################



# Create directory for output data 
data_directory = 'solutions/'

    
# Set the state of the two inlets for each flamelet, iterate through each and solve
ampl = 1.0              # scale mass flow rate of fuel and oxidizer
for ix in range(Nfl):   # iterate through total number of flamelets, Nfl
    mflux = mdot_f*ampl
    f.fuel_inlet.mdot = mflux # set new fuel flow rate inlet
    mflux = mdot_o*ampl
    f.oxidizer_inlet.mdot = mflux # set new oxidizer flow rate inlet
    
    # First disable the energy equation and solve the problem without refining the grid
    f.energy_enabled = False
    f.solve(loglevel = 0, refine_grid=False)
    # Now specify grid refinement, turn on the energy equation, and solve again.
    f.set_refine_criteria(ratio = 200.0, slope = 0.1, curve = 0.2, prune = 0.02) # grid refinement critera
    f.energy_enabled = True
    f.solve(loglevel = 0, refine_grid = True)
    

    
    # Write solution
    nz = f.flame.n_points
    z = f.flame.grid    # grid spacing (m)
    T = f.T             # temperature (K)
    u = f.u             # axial velocity (m/s)
    V = f.V             # Strain (1/s)
    rho = gas.density   # density (kg/m^3)
    
    # Save your results in a CSV file
    file_name1 = 'strain_loop_' + format(ix, '02d') + '.xml'
    file_name2 = 'strain_loop_' + format(ix, '02d') + '.csv'
    f.save(data_directory + file_name1, name='solution', loglevel=1,
                   description='Cantera version ' + ct.__version__ +
                   ', reaction mechanism ' + reaction_mechanism)
    #f.write_csv(data_directory + file_name2, species = 'X' ,quiet=False)
    
    with open(data_directory + file_name2, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['mixF','x (m)','u (m/s)','strain (1/s)','T (K)', 'density (kg/m^3)'] + gas.species_names)
        for n in range(nz):
            Y = gas.Y
            mixtureFraction = f.mixture_fraction('Ar')
            f.set_gas_state(n)
            writer.writerow([mixtureFraction[n],z[n], u[n], V[n], T[n], rho]+list(Y))
            
    # Increment mass flux amplification factor
    ampl = ampl + 0.2

#f.show_solution()








