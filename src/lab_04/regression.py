# Name: Matthew Davidson UCID: 30182729
# File Purpose: Perform multiple linear regression using the normal equation method.
#
#
#
#
import numpy as np
#
#
# Purpose: Perform multiple linear regression using the normal equation method.
#
# Parameters:
#y : array_like, shape = (n,) or (n,1) the vector of dependent variable data
#Z : array_like, shape = (n,m) the matrix of independent variable data

# Returns:
#a : array_like, shape = (m,) or (m,1) the vector of regression coefficients
#residuals : array_like, shape = (n,) or (n,1) the vector of residuals
#R_squared : float the coefficient of determination, a measure of how well the model fits the data
#
def multi_regress(y, Z):
    y = np.asarray(y)
    Z = np.asarray(Z)
    
    # Checking if y is 1D and reshaping it to 2D
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    n = y.shape[0]
    m = Z.shape[1]

    if Z.shape[0] != n:
        raise ValueError("The number of rows in Z must match the number of rows in y.")
    
    Z_transpose_Z = Z.T @ Z
    Z_transpose_Z_inv = np.linalg.inv(Z_transpose_Z)
    Z_transpose_y = Z.T @ y
    a = Z_transpose_Z_inv @ Z_transpose_y
    f = Z @ a
    residuals = y - f

    y_mean = np.mean(y)
    Total_Sum_of_Squares = np.sum((y - y_mean) ** 2)
    Residual_Sum_of_Squares = np.sum(residuals ** 2)
    R_squared = 1 - (Residual_Sum_of_Squares / Total_Sum_of_Squares)

    if y.shape[1] == 1:
        a = a.flatten()
        residuals = residuals.flatten()
    
    return a, residuals, R_squared