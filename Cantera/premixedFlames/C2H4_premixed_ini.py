"""
A stationary premixed ethylene C2H4-air flame with equivalence ratio = 1.
Cantera solution is the steady flame solution which is used to initialize the flame in OpenFOAM as the initial condition.
Laminar flame speed is used to initialize flame and remains constant.
Mixture averaged transport properties.
Chemical mechanisms taken from CHEMKIN files in OpenFOAM case folder for C2H4-air.

Joseph N. Squeo
University of Connecticut
"""

import cantera as ct
import numpy as np
import csv

# Simulation parameters
p = ct.one_atm  # pressure [Pa]
Tin = 300.0  # unburned gas temperature [K]
# reactants = 'C2H4:0.1408, O2:0.1805, N2:0.6787'  # premixed gas mole composition
reactants = 'C2H4:0.0655, O2:0.1963, N2:0.7382'  # [8-3-2020] updated mole fractions based on equiv = 1
width = 0.2  # m
loglevel = 0  # amount of diagnostic output (0 to 8)

# IdealGasMix object used to compute mixture properties, set to the state of the
# upstream fuel-air mixture
# gas = ct.Solution('h2o2.xml')
gas = ct.Solution('chem.cti')
gas.TPX = Tin, p, reactants

# Set up flame object
f = ct.FreeFlame(gas, width=width)
#f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
f.set_refine_criteria(ratio=3, slope=0.2, curve=0.3) #0.01,0.01
f.max_grid_points = 1000
f.show_solution()

# Solve with mixture-averaged transport model
f.transport_model = 'Mix'
f.solve(loglevel=loglevel, auto=True)

# Solve with the energy equation enabled
f.save('c2h4_adiabatic.xml', 'mix', 'solution with mixture-averaged transport')
f.show_solution()
print('mixture-averaged flamespeed = {0:7f} m/s'.format(f.u[0]))
print('max temperature: T_max = {0:7f} K'.format(max(f.T)))
print('T_u = {0:7f} K'.format(f.T[0])) 
print('T_b = {0:7f} K'.format(f.T[-1])) 
maxdT = max(np.gradient(f.T)/np.gradient(f.grid))
print('max(dT/dx) = {0:7f} K/m'.format(maxdT))



# Solve with multi-component transport properties
f.transport_model = 'Multi'
f.solve(loglevel) # don't use 'auto' on subsequent solves
f.show_solution()
print('multicomponent flamespeed = {0:7f} m/s'.format(f.u[0]))
f.save('c2h4_adiabatic.xml','multi', 'solution with multicomponent transport')
print('max temperature: T_max = {0:7f} K'.format(max(f.T)))
print('T_u = {0:7f} K'.format(f.T[0])) 
print('T_b = {0:7f} K'.format(f.T[-1])) 
# maxdT = max(np.gradient(f.T)/np.gradient(f.grid))
# print('max(dT/dx) = {0:7f} K/m'.format(maxdT))

""""
# write the velocity, temperature, density, and mole fractions to a CSV file
f.write_csv('c2h4_adiabatic2.csv', quiet=False)
""""



# Save to .csv file
csv_file = 'c2h4_adiabatic.csv'
with open(csv_file, 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['z (m)'] +  ['U (m/s)'] + ['T (K)'] + gas.species_names)
    for i in range(len(f.grid)):
        writer.writerow([f.grid[i]] + [f.u[i]] + [f.T[i]] + list(f.Y[:,i]))
   
print('Output written to dataOutput.csv')









