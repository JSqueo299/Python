#!/usr/bin/python
#
###############################################################
#
# ADIABATIC_FLAME - A freely-propagating, premixed flat flame 
#             
###############################################################

#import :

from cantera import *
from matplotlib.pylab import *
import numpy as np
#Functions :

#################################################################
# Prepare your run
#################################################################
#Parameter values :
	
	#General
p          =   10000000.0                 # pressure
tin        =   750.0                    # unburned gas temperature
phi        =   1.0


	#Initial grids, Length of the domain has to be adapted to flame thickness = 20-50 *flame_thickness : 
	# - Refined grid at inlet and outlet, 6 points in x-direction :
#initial_grid = 0.5*np.array([0.0, 0.001, 0.01, 0.02, 0.029, 0.03],'d')/3 # m for 1 bar
#initial_grid = 0.05*np.array([0.0, 0.001, 0.01, 0.02, 0.029, 0.03],'d')/3 # m for 10 bars
initial_grid = 0.005*np.array([0.0, 0.001, 0.01, 0.02, 0.029, 0.03],'d')/3 # m for 100 bars

#Set tolerance properties
tol_ss    = [1.0e-5, 1.0e-8]        # [rtol atol] for steady-state problem
tol_ts    = [1.0e-5, 1.0e-8]        # [rtol atol] for time stepping

loglevel  = 1                       # amount of diagnostic output (0
                                    # to 5)
				    
refine_grid = True                  # True to enable refinement, False to
                                    # disable 				   

#Import gas phases with mixture transport model
gas = Solution('12S_H2_Boivin_No_O2.cti','gas')
#################
#Stoechiometry :

fuel_species = 'H2'
m=gas.n_species
stoich_O2 = 0.25*gas.n_atoms(fuel_species,'H')
ifuel = gas.species_index(fuel_species)
io2 = gas.species_index('O2')

x = np.zeros(m,'d')
x[ifuel] = phi
x[io2] = stoich_O2

#################
#Assembling objects :

	#Set gas state to that of the unburned gas
gas.TPX = tin, p, x

	#Create the free laminar premixed flame
f = FreeFlame(gas, initial_grid)
#f.set_fixed_temperature(650)

f.flame.set_steady_tolerances(default=tol_ss)
f.flame.set_transient_tolerances(default=tol_ts)

f.inlet.X = x
f.inlet.T = tin

#set the minimum space step below CANTERA2.1.1 limit
f.set_grid_min(1e-10)

#################################################################
# Program starts here
#################################################################
#First flame:

	#No energy for starters
f.energy_enabled = False

	#Refinement criteria
f.set_refine_criteria(ratio = 7.0, slope = 1, curve = 1)
#f.set_refine_criteria(ratio = 5.0, slope = 0.05, curve = 0.005)

	#Max number of times the Jacobian will be used before it must be re-evaluated
f.set_max_jac_age(50, 50)

	#Set time steps whenever Newton convergence fails
f.set_time_step(5.e-06, [10, 20, 80]) #s

	#Calculation
f.solve(loglevel, refine_grid)

#################
#Second flame:

	#Energy equation enabled
f.energy_enabled = True

	#Refinement criteria when energy equation is enabled
f.set_refine_criteria(ratio = 5.0, slope = 0.5, curve = 0.5)

	#Calculation and save of the results
f.solve(loglevel, refine_grid)

#################
#Third flame and so on ...:

	#Refinement criteria should be changed ...
f.set_refine_criteria(ratio = 5.0, slope = 0.3, curve = 0.3)

f.solve(loglevel, refine_grid)

#################
#Third flame and so on ...:

	#Refinement criteria should be changed ...
f.set_refine_criteria(ratio = 3.0, slope = 0.1, curve = 0.1)

f.solve(loglevel, refine_grid)

################
f.set_refine_criteria(ratio = 2.0, slope = 0.05, curve = 0.05, prune = 0.01)

f.solve(loglevel, refine_grid)

#Fourth flame and so on ...

f.set_refine_criteria(ratio = 2.0, slope = 0.02, curve = 0.02, prune = 0.01)

f.solve(loglevel, refine_grid)

#Fith flame and so on ...

f.set_refine_criteria(ratio = 2.0, slope = 0.01, curve = 0.01, prune = 0.01)

f.solve(loglevel, refine_grid)

#################
#change in initial condition

print 'mixture averaged flamespeed = ',f.u[0]
print 'pressure = ', f.flame.P, 'inlet temperature = ', f.inlet.T

#compute flame thickness
z= f.flame.grid
T = f.T
size = size(z)-1
grad = zeros(size)
for i in range(size):
  grad[i] = (T[i+1]-T[i])/(z[i+1]-z[i])
thickness = (max(T) -min(T)) / max(grad)
print 'laminar flame thickness = ', thickness
print
print 'SCHMIDT NUMBERS (in burnt gases):'
f.set_gas_state(f.flame.n_points-1)
list_species = f.gas.species_names

for species in list_species:
  print species, gas.viscosity/gas.mix_diff_coeffs[gas.species_index(species)]/gas.density

print
print 'PRANDTL NUMBER (in fresh gases):'
f.set_gas_state(f.flame.n_points-f.flame.n_points)
Pr = gas.cp_mass * gas.viscosity / gas.thermal_conductivity
print('Prandtl = '+str(Pr))
print

#save restore_file
f.save('h2_adiabatic.xml', 'energy', 'solution with the energy enabled')

#################################################################
# Save your results if needed
#################################################################
#Write the velocity, temperature, density, and mole fractions to a CSV file
f.write_csv('h2_adiabatic.csv', quiet=False)
f.write_avbp('adiabatic_flame_H2_O2_'+str(tin)+'K_'+str(p*10**-5)+'bar_phi'+str(phi)+'_RM.csv', quiet=False)

#################################################################
# Plot your results
#################################################################
#Plot the velocity, temperature, density

z = f.flame.grid
T = f.T
u = f.u

fig=figure(1)

	# create first subplot - adiabatic flame temperature
a=fig.add_subplot(321)
a.plot(z,T)
title(r'$T_{adiabatic}$ vs. Position')
xlabel(r'Position [m]', fontsize=15)
ylabel("Adiabatic Flame Temperature [K]")
a.xaxis.set_major_locator(MaxNLocator(10)) # this controls the number of tick marks on the axis

	# create second subplot - velocity
b=fig.add_subplot(322)
b.plot(z,u)
title(r'Velocity vs. Position')
xlabel(r'Position [m]', fontsize=15)
ylabel("velocity [m/s]")
b.xaxis.set_major_locator(MaxNLocator(10)) 

        # create third subplot - rho
c=fig.add_subplot(323)
p = np.zeros(f.flame.n_points,'d')
for n in range(f.flame.n_points):
    f.set_gas_state(n)
    p[n]= gas.density_mass
c.plot(z,p)
title(r'Rho vs. Position')
xlabel(r'Position [m]', fontsize=15)
ylabel("Rho [kg/m^3]")
c.xaxis.set_major_locator(MaxNLocator(10)) 

        # create fourth subplot - specie H2
d=fig.add_subplot(324)
h2 = np.zeros(f.flame.n_points,'d')
for n in range(f.flame.n_points):
    f.set_gas_state(n)
    h2[n]= gas.Y[ifuel]
d.plot(z,h2)
title(r'H2 vs. Position')
xlabel(r'Position [m]', fontsize=15)
ylabel("H2 Mole Fraction")
d.xaxis.set_major_locator(MaxNLocator(10))

        # create fith subplot - specie OH
e=fig.add_subplot(325)
ioh = gas.species_index('OH')
oh = np.zeros(f.flame.n_points,'d')
for n in range(f.flame.n_points):
    f.set_gas_state(n)
    oh[n]= gas.Y[ioh]
e.plot(z,oh)
title(r'OH vs. Position')
xlabel(r'Position [m]', fontsize=15)
ylabel("OH Mole Fraction")
d.xaxis.set_major_locator(MaxNLocator(10))

        # Set title
fig.text(0.5,0.95,r'Adiabatic $H_{2}$ + Air Free Flame at Phi = 1 Ti = 300K and P = 1atm',fontsize=22,horizontalalignment='center')

subplots_adjust(left=0.08, right=0.96, wspace=0.25)

show()

f.show_stats


