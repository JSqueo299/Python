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



# ====== FUNCTIONS =====

# Function read .csv files
def readFile(fileToRead):
    fileData = pd.read_csv(fileToRead)
    fileData.head()
    return fileData

def mapFeatureSpace_toNonlinear(X,degree_to_map):
    #Map input feature space to nonlinear space
    featureMapping = PolynomialFeatures(degree_to_map,interaction_only=False)
    X_mapped = featureMapping.fit_transform(X)
    mappedVars=featureMapping.get_feature_names(['T','C2H4','C2H2','O2','O','OH','H2O','CO','CO2','H','H2'])
    return X_mapped,mappedVars

def lasso_regression(X_mapped, Y, data, alpha, models_to_plot={}):
    ppm = 1e6
    #Fit the model
    lassoreg = Lasso(alpha=alpha, normalize=True, max_iter=1e5)
    lassoreg.fit(X_mapped,Y)
    y_pred = lassoreg.predict(X_mapped)
    
    #Check if a plot is to be made for the entered alpha
    T = np.array(data['X'])
    Fv = np.array(Y)

    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(T,y_pred*ppm,'k',label="fit")
        plt.scatter(T,Fv*ppm,facecolors='none',color='salmon',label='training data')
        plt.title('Plot for alpha: %.3g'%alpha)
        plt.xlabel('T (K)')
        plt.ylabel('Fv (ppm)')
        
    
    #Return the result in pre-defined format
    rss = sum((y_pred-Y)**2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret  

       

# ========= READ IN DATASET =========

# Change directory to Somesh Roy flame data
changeDir = '/Users/Joesqueo/OneDrive - University of Connecticut/RESEARCH/Machine_learning/SomeshRoy_premixData/results_commaSepVar'
os.chdir(changeDir)

# Read in column headers (variables)
file = '2EQ-JW1.69-abf31-SNCCN-sp.csv'
data = readFile(file)

# Feature and target space:
X = data[['T','C2H4','C2H2','O2','O','OH','H2O','CO','CO2','H','H2']]
Y = data['Fv']
X_array = np.array(X)
numVars = len(X_array[0])

# Test/train data split using cross-validation
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)
X_train_orig = X_train
X_test_orig = X_test



# ========= TEST PENALTY COEFF. =========

#Define the alpha values to test
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]
numAlphas = len(alpha_lasso)

#Define the models to plot
models_to_plot = {1e-15:231, 1e-10:232, 1e-5:233, 1e-3:234, 1e-2:235, 1:236}

#Map linear feature space to nonlinear feature space
degree_to_map = 3
X_mapped,mappedVars = mapFeatureSpace_toNonlinear(X,degree_to_map)

#Initialize the dataframe to store coefficients
#col = ['MSE','intercept'] + ['coef_%d'%i for i in range(1,len(X_mapped[0])+1)]
col = ['MSE','intercept'] + mappedVars
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,numAlphas)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)   

#Iterate over the 10 alpha values:
for i in range(numAlphas):
    coef_matrix_lasso.iloc[i,] = lasso_regression(X_mapped, Y, data, alpha_lasso[i], models_to_plot)

plt.show()

#Save regression coefficients to an excel spreadsheet
os.chdir('/Users/Joesqueo/OneDrive - University of Connecticut/RESEARCH/Python/machineLearning')
coef_matrix_lasso.to_excel("coeffs_outputRAW.xlsx")  