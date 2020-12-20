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
from sklearn.model_selection import train_test_split
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

# Feature and targer space:
X = data[['X','T','C2H4','C2H2','O2','O','OH','H2O','CO','CO2','H','H2']]
Y = data['Fv']

# Test/train data split using cross-validation
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)
X_train_orig = X_train
X_test_orig = X_test

# PRINCIPLE COMPONENT ANALYSIS:
# 1. Import standarizing and PCA libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()

# 2. Standardizing the features
# Fit on training dataset only
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 3. Principle component analysis on feature space 
pca = PCA(.98) #PCA(n_components=5)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
columns = ['principal comp. 1:  ', 'principal comp. 2:  ','principal comp. 3:   ','principal comp. 4:   ','principal comp. 5:   ']

#principalComponents = pca.fit_transform(X_train)
#principalDf = pd.DataFrame(data = principalComponents, columns = ['principal comp. 1', 'principal comp. 2','principal comp. 3','principal comp. 4','principal comp. 5'])
#print(principalDf)
#print(pca.explained_variance_ratio_)

# Regression fit:
regr = linear_model.LinearRegression()

# Train the model, print the coefficient values:
regr.fit(X_train,Y_train)
coeff_data = pd.DataFrame(regr.coef_ , columns[0:len(X_train[0,:])] , columns=['Coefficients'])
print(coeff_data)

# Now let’s do prediction of data:
Y_pred = regr.predict(X_test)

# Check accuracy:
from sklearn.metrics import r2_score
R = r2_score(Y_test , Y_pred)
print ('R² :',R)
print('\n')

xTrain = np.array(X_train_orig['X'])
xTest = np.array(X_test_orig['X'])
yTrain = np.array(Y_train)

# Plotting the regression line:
plt.scatter(xTrain, yTrain*(1e6), color='blue',label="training data")
plt.scatter(xTest, Y_pred*(1e6), color='red',label="prediction")
plt.xlabel("distance (m)")
plt.ylabel("soot Fv (ppm)")
plt.title("Linear MultiVar Regression")
plt.legend()
plt.show()
