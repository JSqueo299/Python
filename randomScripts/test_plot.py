import cantera as ct
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

## Extinction strain rates
# First entry is for unperturbed case
K_ext = [939.532, 786.00, 882.539, 902.186, 850.255, 941.919, 1008.677, 1066.211, 1011.393, 800.434, 897.29, 773.17, 763.485, 1147.845, 840.295, 897.243, 922.103, 790.649, 798.221, 
         847.61, 830.177, 616.86, 911.122, 894.87, 913.582, 890.86, 876.32, 850.003, 816.803, 812.744]
speed = [0.1795, 0.15527, 0.1698, 0.1674, 0.1670, 0.180729, 0.1813, 0.1980, 0.1848, 0.1570, 0.1716, 0.1533, 0.1548, 0.2019, 0.1639, 0.1764, 0.1748, 0.1571, 0.1596, 
         0.1665, 0.1599, 0.1256, 0.17936, 0.17172, 0.17817, 0.1652, 0.1703, 0.1643, 0.1657, 0.1574]


unperturbed = [0.1735, 981.78]


## Plotting
#plt.scatter(S_L, K_ext, s=25, facecolors='none', edgecolors='b', label='Perturbed')
plt.scatter(speed, K_ext, s=25, facecolors='none', edgecolors='b', label='Perturbed')
plt.scatter(unperturbed[0], unperturbed[1], s=25, facecolors='none', edgecolors='r', label='Nominal')
plt.title('Relationship between extinction strain rate and laminar flame speed')
plt.xlabel('Laminar Flame Speed, m/s')
plt.ylabel('Extinction Strain Rate, 1/s')
plt.legend(loc='upper left')
plt.show()

	
