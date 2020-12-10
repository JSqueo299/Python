%reset

"""
Simulates 2 opposed premixed jets, with a strained flame inbetween.  Iterates strain rate by stepping mass flux up at inlet
"""

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time

start=time.time()
currentDT = datetime.datetime.now()
print('Start Time:')
print (str(currentDT))
print('\n\n')

# Differentiation function for data that has variable grid spacing Used here to
# compute normal strain-rate
def derivative(x, y):
    dydx = np.zeros(y.shape, y.dtype.type)

    dx = np.diff(x)
    dy = np.diff(y)
    dydx[0:-1] = dy/dx

    dydx[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])

    return dydx

def computeStrainRates(oppFlame):
    # Compute the derivative of axial velocity to obtain normal strain rate
    strainRates = derivative(oppFlame.grid, oppFlame.u)

    #Find maxium positive strain location (occurs in flame)
    #Identify minimum velocity point in upstream direction
    #This is the start of the preheat zone
    maxStrLocation = strainRates.argmax()
    if maxStrLocation==0:
        maxStrLocation=abs(strainRates).argmax()
    minVelocityPoint = oppFlame.u[:maxStrLocation].argmin()

    # Characteristic Strain Rate = K
    # Strain reported is maximum strain before the preheat zone
    strainRatePoint = abs(strainRates[:minVelocityPoint]).argmax()
    K = abs(strainRates[strainRatePoint])

    return strainRates, strainRatePoint, K

# Create directory for output data files
"""
==========================================
==SET DIRECTORY TO SAVE RESTART FILES HERE==
==========================================
"""
data_directory = 'skeletal_25_extinction_tests/'
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
    
#Set input parameters
P=ct.one_atm
T=300 #K
phi=0.7
width=0.03 #meters
loglevel=0


#Generate gas object
"""
===============================
==SET REACTION MECHANISM HERE==
===============================
"""
reaction_mechanism="FFCM-1.cti"
gas=ct.Solution(reaction_mechanism) #25 species skeletal mechanism
gas.TPX=T,P,{'CH4':phi,'O2':2,'N2':3.76*2}
gas()
rho=gas.density

# Domain half-width of 2.5 cm, meaning the whole domain is 5 cm wide
width = 0.025




# Create the flame object
f = ct.CounterflowTwinPremixedFlame(gas, width=width)
ratio=3
slope=0.05
curve=0.05
prune=slope*0.1
GP_max=10**6

f.set_refine_criteria(ratio=ratio, slope=slope,curve=curve,prune=prune)
f.max_grid_points=GP_max

print('Blue = T')
print('Red = Normal Strain')
print('Black = Velocity')
print('Red x = Characterisitic Strain Location')
print('Black . = Grid Points')

strain_list=list()
temp_list=list()

#Set initial and final inlet velocity, and desired increments
u_i=1 #m/s
du=1 #m/s
u=u_i-du
i=0
dKdT=0

#file_name = 'extinction_{0:08f}.xml'.format(u)
#f.restore(data_directory + file_name, name='solution', loglevel=0)
calc1=time.time()
calc2=0

while True:

    print('\n\n##############################################################')
    i+=1
    print('Trial # %d' %(i))
    u+=du
    massFlux=rho*u
    f.reactants.mdot=massFlux
    f.set_refine_criteria(ratio=ratio, slope=slope,curve=curve,prune=prune)
    f.solve(loglevel=loglevel,auto=True)
    (strainRates, strainRatePoint,K)=computeStrainRates(f)
    calc2=calc1
    calc1=time.time()
    print('Calculation Time = %8.1f min' %((calc1-calc2)/60))
    print('Inlet Velocity = %8.6f m/s' %(u))
    print('Velocity Step = %8.6f m/s' %(du))
    print('Slope = %8.6f' %(slope))
    print('Number of GP = %6d' %(len(f.grid)))
    if max(f.T)>500:
        strain_list.append(K)
        temp_list.append(max(f.T))
        print('Characteristic strain = %8.6f 1/s' %(K))
        print('T_max = %8.4f K' %(max(f.T)))
        if len(strain_list)>1:
            dKdT=(strain_list[-1]-strain_list[-2])/(temp_list[-1]-temp_list[-2])
            print('dK/dT = %8.6f (1/s)/K' %(dKdT))                
        file_name = 'extinction_{0:08f}.xml'.format(u)
        f.save(data_directory + file_name, name='solution', loglevel=0,
               description='Cantera version ' + ct.__version__ +
               ', reaction mechanism ' + reaction_mechanism)
        
        plt.plot(f.grid,f.T/max(f.T),'b-',f.grid,strainRates/max(strainRates),'r-',f.grid,f.u/max(f.u),'k-',f.grid[strainRatePoint],-K/max(strainRates),'rx')
        plt.xlabel('x')
        plt.title('Inlet velocity = %8.6f m/s' %(u))
        plt.plot(f.grid,[0]*len(f.grid),'k.')
        plt.show()
        if len(strain_list)>1 and dKdT>-1:
            print('Slope of extinction curve has reached turning point critera')
            print('dK/dT>-1')
            print('Terminating')
            break
    elif du>0.05:
        print('Flame extinguished')
        print('Attempt refining step size')
        file_name = 'extinction_{0:08f}.xml'.format(u-du)
        f.restore(data_directory + file_name, name='solution', loglevel=0)
        u-=du
        du=du/2
    elif slope>0.001:
        print('Flame extinguished')
        print('Attempt refining grid')
        file_name = 'extinction_{0:08f}.xml'.format(u-du)
        f.restore(data_directory + file_name, name='solution', loglevel=0)
        if slope<0.005:
            ratio=2.5            
        slope=slope/2
        curve=slope
        prune=slope/10
        u-=du
    elif du>0.00005:
        print('Flame extinguished')
        print('Attempt refining step size')
        file_name = 'extinction_{0:08f}.xml'.format(u-du)
        f.restore(data_directory + file_name, name='solution', loglevel=0)
        u-=du
        du=du/2
    else:
        print('Exceeded step size and grid refinement')
        print('Terminating')
        print('##############################################################\n\n')
        break
    print('##############################################################\n\n')
        



    
print('\n\n##############################################################')
print('Run Complete\n')
print('Extinction strain rate = %8.6f 1/s\n\n' %(max(strain_list)))
plt.plot(strain_list,temp_list)
plt.xlabel('Strain 1/s')
plt.ylabel('Temp K')
plt.title('Extinction Strain Rate')
plt.show()
print('\n\n##############################################################')

currentDT = datetime.datetime.now()
print('End Time:')
print (str(currentDT))
print('\n\n')
end=time.time()
print('Total Elapsed Time: %8.1f Hours' %((end-start)/3600))