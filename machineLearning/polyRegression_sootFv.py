#################################################################################
# Machine Learning Polynomial Regression for Soot Fv prediction
# Joseph N. Squeo
# joseph.squeo@uconn.edu
# 12-9-2020
#
# DESCRIPTION:
#   Polynomial regression machine learning package in Python is used to build a nonlinear
#   model to predict soot volume fraction. Training data is 1D laminar premixed flames 
#   supplied by Somesh Roy
#################################################################################

# Import the required libraries:
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use("TkAgg")
from sklearn import linear_model

# Read the data file:
data = np.loadtxt( './Results/2EQ-JW1.69-DLR-SNCCN-sp.dat',skiprows=1)
#print(data[0,1])
T = data[:,1]
Y_C2H2 = data[:,7]
Y_C3H6 = data[:,11]
Y_C2H4 = data[:,8]
Y_OH = data[:,39]
Y_O = data[:,38]
Y_O2 = data[:,15]
Y_C2 = data[:,16]
Y_C = data[:,24]
Y_CO = data[:,19]
Y_CO2 = data[:,20]
Y_H = data[:,25]
Y_H2O = data[:,17]
Y_H2 = data[:,5]
Fv = data[:,103]

# Consider features we want to work on:
X = [T,Y_C2H2,Y_C3H6,Y_C2H4,Y_OH,Y_O,Y_O2,Y_C2,Y_C,Y_CO,Y_CO2,Y_H,Y_H2O,Y_H2]
#X = data[[ ‘ENGINESIZE’, ‘CYLINDERS’, ‘FUELCONSUMPTION_CITY’,’FUELCONSUMPTION_HWY’, ]
Y = [Fv, T]

print('Y is \n',Y)





