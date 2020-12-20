#################################################################################
# Script used to import VTK data from OpenFOAM to Python/Cantera to ignite the flame. Use foamToVTK in OpenFOAM to convert data to VTK. Script reads in data from OpenFOAM, then solves a diffusion flame at equilibrium for all cells in the mesh, which is saved into a csv file.
# Download pyvista at the following link: https://pswpswpsw.github.io/posts/2018/09/blog-post-modify-vtk-openfoam/
# Or you can use:    sudo pip install pyvista
#
# Joseph N. Squeo
# 5-1-2020
#################################################################################
import cantera as ct
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pyvista as vtki
import csv

## grid is the central object in VTK where every field is added on to grid
grid = vtki.UnstructuredGrid('./VTK/scratch_linearTprofile2x_4133540.vtk')

## point-wise information of geometry is contained
#print(grid.points)

## get a dictionary contains all cell/point information
#print(grid.cell_arrays) # note that cell-based and point-based are in different size
#print(grid.point_arrays) # 

## get a field in numpy array
N = grid.cell_arrays['cellID']
T = grid.cell_arrays['T']
C2H4 = grid.cell_arrays['C2H4']
O2 = grid.cell_arrays['O2']
N2 = grid.cell_arrays['N2']

# Save to .csv file (before equilibrium)
csv_file = 'beforeEquilib.csv'
with open(csv_file, 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['N'] + ['T'] + ['Y_C2H4'])
    for i in range(len(N)):
        writer.writerow([N[i]] + [T[i]] + [C2H4[i]])



## Create gas object and allocate space
gas = ct.Solution('chem.cti')
T_ad = np.zeros(len(N))
y = np.zeros(shape = (len(N),32))


for i in range(len(N)): 
    if C2H4[i] < 0.95:
        gas.TPY = T[i], 101325,{'C2H4':C2H4[i], 'O2':O2[i], 'N2':N2[i]}
        gas.equilibrate('HP')
        y[i,:] = gas.Y
        T_ad[i] = gas.T
    else:
        T_ad[i] = T[i] 
        y[i,21] = C2H4[i]   # Index is 22-1 since python array start at i=0
        y[i,3] = O2[i]
        y[i,31] = N2[i]

# Save to .csv file
csv_file = 'dataOutput.csv'
with open(csv_file, 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['T'] + gas.species_names)
    for i in range(len(N)):
        writer.writerow([T_ad[i]] + list(y[i]))
   
print('Output written to dataOutput.csv')
    


#for i, specie in enumerate(gas.species()):
#    print(str(i) + '. ' + str(specie))

