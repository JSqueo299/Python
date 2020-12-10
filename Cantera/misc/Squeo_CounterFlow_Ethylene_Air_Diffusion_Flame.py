# ============================================================================================= #
from __future__ import print_function
from __future__ import division

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

from cantera import *
from numpy import *

print("Running Cantera Version: " + str(ct.__version__))
print("\nAn opposed-flow ethylene/air diffusion flame")
# ============================================================================================= #



# Differentiation function for data that has variable grid spacing Used here to
# compute normal strain-rate
def derivative(x, y):
    dydx = np.zeros(y.shape, y.dtype.type)

    # The first difference is given by out[n] = a[n+1] - a[n] along the given axis, higher differences are calculated
    # by using diff recursively.
    dx = np.diff(x)
    dy = np.diff(y)
    # Create an array starting at index [0] of dy and dx and span through to the last value given by [-1]
    dydx[0:-1] = dy/dx

    # Backwards finite difference for last term in dydx array since no point for forward difference
    dydx[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])

    return dydx

def computeStrainRates(f):
    # Compute the derivative of axial velocity to obtain normal strain rate
    strainRates = derivative(f.grid, f.u)

    # Obtain the location of the max. strain rate upstream of the pre-heat zone.
    # This is the characteristic strain rate
    maxStrLocation = abs(strainRates).argmax()
    minVelocityPoint = f.u[:maxStrLocation].argmin()

    # Characteristic Strain Rate = K
    strainRatePoint = abs(strainRates[:minVelocityPoint]).argmax()
    K = abs(strainRates[strainRatePoint])
    
    # Max Strain Rate = a_max
    a_max = max(abs(strainRates))

    return strainRates, strainRatePoint, K, a_max


#Create the gas object, GRI3.0 is optimized chemical combustion mechanism
gas = ct.Solution('GRI_mech_cti.cti')

#Parameter values :
p = 1e5 # pressure, Pa

#Input Parameters
T_in_f = 300.0 # fuel inlet temperature, K
T_in_o = 300.0 # oxidizer inlet temperature, K
mdot_f = .24 # fuel inlet kg/m^2/s
mdot_o = .72 # oxidizer inlet kg/m^2/s
comp_f = 'C2H4:1' # fuel composition
comp_o = 'O2:0.21, N2:0.78, AR:0.01' # air composition

#Set gas state for the flame domain
gas.TP = T_in_o, p

#Create an evenly-spaced 20 point grid that has a 2cm distance between the burners 
# Note when the solver runs, the mesh is changed due to mesh refinement and addition of new points
width = 0.020 # width between burners = 2cm
N = 20 # Number of grid points in domain
initial_grid = linspace(0, width, N)
dx = width/N #Spacing between grid points

#Cantera auto-refines grid (adds grid points) by solving non-reacting continutity then the energy equation. #Therefore, the initial grid essentially does not matter, it is only required to start the simulation






# Vary strain and plot T to see affect
#Initial Strain

#Density of oxidizer and fuel
rho_o = p/(8.314/0.029*T_in_o)
rho_f = p/(8.314/0.030*T_in_f)

#Velocity from mass flow rates
vel_o = mdot_o/rho_o
vel_f = mdot_f/rho_f


# Create an object representing the counterflow flame configuration,
# which consists of a fuel inlet on the left, the flow in the middle,
# and the oxidizer inlet on the right.
f = CounterflowDiffusionFlame(gas, initial_grid)

# Set the state of the two inlets
f.fuel_inlet.mdot = mdot_f
f.fuel_inlet.X = comp_f
f.fuel_inlet.T = T_in_f

f.oxidizer_inlet.mdot = mdot_o
f.oxidizer_inlet.X = comp_o
f.oxidizer_inlet.T = T_in_o

# Set the boundary emissivities
f.set_boundary_emissivities(1, 1) #An optically-thin radiation model for CO2 and H2O
# Turn radiation off
f.radiation_enabled = False

f.set_refine_criteria(ratio=4, slope=0.2, curve=0.3, prune=0.04) # grid refinement critera

#Set error tolerances
tol_ss = [1.0e-5, 1.0e-11] # [rtol, atol] for steady-state problem
tol_ts = [1.0e-5, 1.0e-11] # [rtol, atol] for time stepping
f.flame.set_steady_tolerances(default=tol_ss)
f.flame.set_transient_tolerances(default=tol_ts)

# Solve the problem
f.solve(loglevel=0, auto=True)
# f.show_solution()
#f .show_stats(0)











# Plot Temperature without radiation
# Import plotting modules and define plotting preference
#import matplotlib.pyplot as plt
#matplotlib notebook

plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['figure.figsize'] = (8,6)

plt.figure()
plt.plot(f.flame.grid, f.T, label='Temperature without radiation')
plt.title('Temperature of the flame')
plt.ylim(0,2500)
plt.xlim(0.000, 0.020)
plt.show()

# Turn on radiation and solve again
f.radiation_enabled = True
f.solve(loglevel=1, refine_grid=False)
f.show_solution()

# Plot Temperature with radiation
plt.plot(f.flame.grid, f.T, label='Temperature with radiation')
plt.legend()
plt.legend(loc=2)
plt.title('Temperature v. Axial Distance')
plt.xlabel('Axial Distance (cm)')
plt.ylabel('Temperature (K)');
plt.grid()










################### The solver starts here###################
#First flame:

#disable the energy equation
f.energy_enabled = False

# Set error tolerances
tol_ss = [1.0e-5, 1.0e-11] # [rtol, atol] for steady-state problem
tol_ts = [1.0e-5, 1.0e-11] # [rtol, atol] for time stepping
f.flame.set_steady_tolerances(default=tol_ss)
f.flame.set_transient_tolerances(default=tol_ts)

#solve the problem WITHOUT refining the grid
f.solve(loglevel = 0, refine_grid=False)


#Second flame:

# specify grid refinement criteria, turn on the energy equation, allow grid refinement and 
# specify when grid points added to slope and curve
f.energy_enabled = True
f.set_refine_criteria(ratio=3, slope=0.2, curve=0.3, prune=0.04)

######REFINEMENT CRITERIA########
#ratio : Additional points will be added if the ratio of the spacing on either side
#of a grid point exceeds this value

#slope : Maximum difference in value between two adjacent points, scaled by the
#maximum difference in the profile (0.0 < slope < 1.0).
#Adds points in regions of high slope.

#curve : Maximum difference in slope between two adjacent intervals, scaled by the
#maximum difference in the profile (0.0 < curve < 1.0).
#Adds points in regions of high curvature

#prune: Removes grid points
#################################

#solve the problem again
f.solve(loglevel = 1, refine_grid = True)

#save your results
f.save('C2H4_diffusion.xml','energy')
# write the velocity, temperature, and mole fractions to a CSV file
f.write_csv('C2H4_diffusion.csv', quiet=False)


# Import plotting modules and define plotting preference
import matplotlib.pyplot as plt
#matplotlib notebook

plt.rcParams['figure.autolayout'] = True

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['figure.figsize'] = (8,6)


# Axial Velocit Plot
plt.figure()
plt.show()
plt.plot(f.grid*100, f.u*100, 'r-o', lw=2)
plt.plot(f.grid*100,np.zeros(size(f.grid)),'--', lw=2)
plt.xlim(f.grid[0], f.grid[-1]*100)
plt.title('Axial Velocity Plot')
plt.xlabel('Axial Distance (cm)')
plt.ylabel('Axial Velocity (cm/s)')
plt.grid()



# Temperature Plot
plt.figure()
plt.show()
plt.plot(f.grid*100, f.T, 'b-s', lw=2)
plt.xlim(f.grid[0], f.grid[-1]*100)
plt.title('Temperature v. Axial Distance')
plt.xlabel('Axial Distance (cm)')
plt.ylabel('Temperature (K)');
maxT = max(f.T)
minT = min(f.T)
print('Max Temperature =',maxT)
print('Min Temperature =',minT)
plt.grid()
Tmax_index = np.argmax(f.T) # Index of the max T in the f.T array
plt.plot(f.grid[Tmax_index]*100,maxT,'r-s', lw=5) # plot max T location


# To plot species, we first have to identify the index of the species in the array
# For this, cut & paste the following lines and run in a new cell to get the index
for i, specie in enumerate(gas.species()):
    print(str(i) + '. ' + str(specie))
	
	
# Extract concentration data
X_C2H4 = f.X[24]
X_CO2 = f.X[15]
X_H2O = f.X[5]
X_N2 = f.X[47]
X_O2 = f.X[3]    

plt.figure()
plt.show()
plt.plot(f.grid*100, X_C2H4, 'c-o', lw=2, label=r'$C_{2}H_{4}$')
plt.plot(f.grid*100, X_CO2, 'm->', lw=2, label=r'$CO_{2}$')
plt.plot(f.grid*100, X_H2O, 'g->', lw=2, label=r'$H_{2}O$')
plt.plot(f.grid*100, X_O2, 'r->', lw=2, label=r'$O_{2}$')
plt.plot(f.grid*100, X_N2, 'b->', lw=2, label=r'$N_{2}$')

plt.xlim(f.grid[0], f.grid[-1]*100)
plt.title('Mole Fraction Composition v. Axial Distance')
plt.xlabel('Axial Distance (cm)')
plt.ylabel('Mole Fractions')

plt.legend(loc=2);
	
	
	
	
#The negative index refers to starting at the last index [-1] and work backwards. Remeber in python index starts at 0
# which is the first point and proceeds either forwards [1], [2]... or backwards starting at last point [-1],[-2]...

# STRAIN RATE DEINITIONS: https://www.ems.psu.edu/~radovic/ChemKinTutorials_PaSR.pdf page 94 of pdf

# GLOBAL (MEAN) STRAIN RATE:
# Global strain rate =  difference between oxidizer and fuel velocities divided by distance between inlets
# Note that oxidizer inlet velocity is negative, fuel is positive since flowing in opposite directions
My_global_strain_rate = (vel_f + vel_o)/width    # s-1
print("Global strain rate = "+str(My_global_strain_rate)+" s^-1")

# Utilizes built in strain rate function of counterflow solver to calculate the mean axial velocity gradient 
# in the entire domain ----> (self.u[-1] - self.u[0]) / self.grid[-1]. 
#
# f.u[-1] = last point in flame axial velocity array across 1D mesh(oxidizer inlet)
# f.u[0] = first point in flame axial velocity array across 1D mesh(fuel inlet)
# f.grid[-1] = last point in grid array, which spans 0 to 2cm, so it is 2cm or the width between inlets
mean_strain_rate1 = f.strain_rate('mean', fuel=None, oxidizer=None,stoich=None) # s^-1
print("Mean strain rate = "+str(mean_strain_rate1)+" s^-1")
mean_strain_rate2 = (f.u[0]-f.u[-1])/0.02
print("Mean strain rate 2 = "+str(mean_strain_rate2)+" s^-1")


# EXTINCTION (MAX) STRAIN RATE:
# Extinction Strain Rate = maximum axial velocity gradient on the fuel side just before the flame

# My max extinction strain rate
# diff uses first order forward difference
My_Ext_strain_rate=max(abs(diff(f.u)/diff(f.grid)))
print("\nMy Extinction strain rate = "+str(My_Ext_strain_rate)+" s^-1")

# np.max(np.abs(np.gradient(self.u) / np.gradient(self.grid)))
Ext_strain_rate = f.strain_rate('max', fuel=None, oxidizer=None,stoich=None)  # s^-1
print("Extinction strain rate = "+str(Ext_strain_rate)+" s^-1")


# USER DEFINED FUNCTION FOR MAX STRAIN RATE  (see 2nd cell above for function)
(strainRates, strainRatePoint, K, a_max) = computeStrainRates(f)
print("Max strain rate (function defined) = "+str(a_max)+" s^-1")



# CANTERA OUTPUT f.V
Max_strain_rate = max(abs(f.V))
print("\nMaximum strain rate = "+str(Max_strain_rate)+" s^-1")
Min_strain_rate = min(f.V)
print("\nMinimum strain rate = "+str(Min_strain_rate)+" s^-1")

#The gradient is computed using second order accurate central differences in the interior points
#and either first or second order accurate one-sides (forward or backwards) differences at the boundaries.
#The returned gradient hence has the same shape as the input array.
# np.gradient(f.u) takes central difference of interior points of f.u array divided by 2 (distance of cells)
# Specifying np.gradient(f.u,x) uses values in array x as spacing, so central diff/value in f.grid
g = abs(np.gradient(f.u)/np.gradient(f.grid))







#Define fuel and oxidizer as species index to reference during mass fraction
Fuel = gas.species_index('C2H4')
Oxidizer = gas.species_index('O2')

# Extract mass fraction data
Y_C2H4 = f.Y[Fuel]
Y_O2 = f.Y[Oxidizer]

# MASS FRACTION v. MIXTURE FRACTION PLOT
plt.figure()
plt.show()
# Mixture fraction is essentially the elemental mass fraction that originates at the fuel stream inlet
# The mixture fraction is computed from the elemental mass fraction of element m, normalized by its values
# on the fuel and oxidizer inlets:
plt.plot(f.mixture_fraction('C'), Y_C2H4, 'c->', lw=2, label=r'$Fuel (C_{2}H_{4})$')
plt.plot(f.mixture_fraction('C'), Y_O2, 'r->', lw=2, label=r'$Oxidizer (O_{2})$')

plt.title('Mass Fraction vs. Mixture Fraction of Carbon')
plt.xlabel('Mixture Fraction, Z')
plt.ylabel('Mass Fraction')

plt.legend(loc=2);


# TEMPERATURE v. MIXTURE FRACTION PLOT
plt.figure()
plt.plot(f.mixture_fraction('C'), f.T, 'b-o',lw=2)
plt.title('Temperature vs Mixture Fraction of C')
plt.xlabel('Mixture Fraction, Z')
plt.ylabel('Temperature (K)')
plt.show()



# Methods to output mass fraction
    # gas['CH4', 'O2'].Y
    # f.Y[species_index_num, :] where : indicates all time steps in mesh