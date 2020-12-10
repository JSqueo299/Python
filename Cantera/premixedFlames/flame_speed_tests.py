import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

phi_rng = np.linspace(0.3,3,28)
sl=[]
T=[]
for phi in phi_rng:
    # Simulation parameters
    p = 3.0 * ct.one_atm  # pressure [Pa]
    Tin = 850.0  # unburned gas temperature [K]
    reactants = 'H2:%.2f, O2:0.5' %(phi)  # premixed gas composition
    width = 0.03  # m
    loglevel = 0  # amount of diagnostic output (0 to 8)

    # IdealGasMix object used to compute mixture properties, set to the state of the
    # upstream fuel-air mixture
    gas = ct.Solution('gri30.cti')
    gas.TPX = Tin, p, reactants

    # Set up flame object
    f = ct.FreeFlame(gas, width=width)
    f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)


    # Solve with mixture-averaged transport model
    f.transport_model = 'Mix'
    f.solve(loglevel=loglevel, auto=True)
    #f.show_solution()
    print('phi = %.2f' %(phi))
    print('mixture-averaged flamespeed = {0:7f} m/s\n'.format(f.u[0]))
    sl=np.append(sl,f.u[0])
    T=np.append(T,np.max(f.T))
    plt.plot(f.grid,f.T)
    plt.show()
    
    
plt.figure()
plt.plot(phi_rng,sl)
plt.show()

plt.figure()
plt.plot(phi_rng,T)
plt.show()

