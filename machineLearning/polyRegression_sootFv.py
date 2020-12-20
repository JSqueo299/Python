#################################################################################
# Machine Learning Multi-Var Linear Regression for Soot Fv prediction
# Joseph N. Squeo
# joseph.squeo@uconn.edu
# 12-9-2020
#
# DESCRIPTION:
#   Polynomial regression machine learning package in Python is used to build a nonlinear
#   model to predict soot volume fraction. Training data is 1D laminar premixed flames 
#   supplied by Somesh Roy.
#
# INPUTS:   
#   Raw .dat files from Somesh Roy need to be converted to .csv files in Excel. Open Excel,
#   click >File > Open, then right click .dat file to 'rename' then the Open button is highlighted
#   blue, so open the file. Fix the variable header row then save as .csv file in the right folder.
#################################################################################

# Import the required libraries:
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn import linear_model
import os

# Function read .csv files
def readFile(fileToRead):
    fileData = pd.read_csv(fileToRead)
    fileData.head()
    return fileData


# Change directory to Somesh Roy flame data
changeDir = '/Users/Joesqueo/OneDrive - University of Connecticut/RESEARCH/Machine_learning/SomeshRoy_premixData/results_commaSepVar'
os.chdir(changeDir)

# Read in column headers (variables)
file = '2EQ-JW1.69-abf31-SNCCN-sp.csv'
data = readFile(file)

# Consider features we want to work on:
X = data[['T','C2H4','C2H2','O2','O','OH','H2O','CO','CO2','H','H2']]
Y = data['Fv']

# Generating training and testing data from our data:
# We are using 80% data for training.
train = data[:(int((len(data)*0.8)))]
test = data[(int((len(data)*0.8))):]

# Modeling:
# Use sklearn package to model data :
regr = linear_model.LinearRegression()

train_X = np.array(train[['T','C2H4','C2H2','O2','O','OH','H2O','CO','CO2','H','H2']])
train_Y = np.array(train['Fv'])

regr.fit(train_X,train_Y)

test_X = np.array(test[['T','C2H4','C2H2','O2','O','OH','H2O','CO','CO2','H','H2']])
test_Y = np.array(test['Fv'])

# Print the coefficient values:
coeff_data = pd.DataFrame(regr.coef_ , X.columns , columns=['Coefficients'])
print(coeff_data)

# Now let’s do prediction of data:
Y_pred = regr.predict(test_X)

# Check accuracy:
from sklearn.metrics import r2_score
R = r2_score(test_Y , Y_pred)
print ('R² :',R)

print('\n')
x = np.array(test['X'])
Fv_test = np.array(test['Fv'])
#dataID = list(range(1,len(Fv_test)+1))

# Plotting the regression line:
plt.scatter(x, Fv_test*(1e6), color='blue')
plt.scatter(x, Y_pred*(1e6), color='red')
plt.xlabel("distance (m)")
plt.ylabel("soot Fv (ppm)")
plt.show()

print(train('Fv'))
