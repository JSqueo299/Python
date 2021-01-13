#################################################################################
# Machine Learning Multi-Var Non-Linear Regression for Soot Fv prediction
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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
import os

# ===== FUNCTIONS =====
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

# Feature and target space:
X = data[['T','C2H4','C2H2','O2','O','OH','H2O','CO','CO2','H','H2']]
Y = data['Fv']

# Test/train data split using cross-validation
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)
X_train_orig = X_train
X_test_orig = X_test
'''
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
columns = ['principal comp. 1:  ', 'principal comp. 2:  ','principal comp. 3:   ','principal comp. 4:   ','principal comp. 5:   ','principal comp. 6:   ','principal comp. 7:   ','principal comp. 8:   ']
#print(pca.explained_variance_ratio_)
'''

# ========= TEST POLYNOMIAL DEGREE =========
# Alpha (regularization strength) of LASSO regression
lasso_eps = 0.001             # eps = alpha_min/alpha_max 
lasso_nalpha = 20           # number of alphas to test
lasso_iter = 5000           # the maximum number of iterations

# Min and max degree of polynomials features to consider
degree_min = 2
degree_max = 8

# Make a pipeline model with polynomial transformation and LASSO regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)
for degree in range(degree_min,degree_max+1):
    model = make_pipeline(PolynomialFeatures(degree, interaction_only=False), LassoCV(eps=lasso_eps,n_alphas=lasso_nalpha,max_iter=lasso_iter,normalize=True,cv=5))
    model.fit(X_train,Y_train)
    test_pred = np.array(model.predict(X_test))
    RMSE = np.sqrt(np.sum(np.square( (test_pred-Y_test)*(1e6) )))     # sum[ (y_test - y_pred)^2 ]
    test_score = model.score(X_test,Y_test)                 # R^2 coefficient of the model
    
    # Plot R^2 vs. polynomial degree:
    plt.subplot(121)
    plt.scatter(degree, test_score, color='blue',label="training data")
    plt.xlabel("model complexity (degree of polynomial)")
    plt.ylabel("R^2 coefficient")

    # Plot RMSE vs. polynomial degree:
    plt.subplot(122)
    plt.scatter(degree, RMSE, color='red',label="prediction")
    plt.xlabel("model complexity (degree of polynomial)")
    plt.ylabel("RMSE (fv, ppm)")
    
plt.show()

