"""
A file to compute and save the FPR
"""

# Import statements
import numpy as np
import pyrft as pr


X = pr.groupX(categ); 

# Initialize the contrast matrix
C = np.array([[1,-1,0],[0,1,-1]]); 

nsubj_vec = np.arange(10,101,10)
dim_sides = np.array([1,5,10,25,50])

# Initialize matrices to store the estimated FPRs
store_JER = zeros((len(nsubj_vec), len(dim_sides)))
store_FWER = store_JER

# Choose the smoothness, the number of bootstraps and the number of iterations to use
FWHM = 0; B = 100; niters = 5000
for J in np.arange(dim_sides):
  print('J:' J)
  Dim = (dim_sides[J], dim_sides[J])
  for I in np.arange(len(nsubj_vec)):
    print('I:', I)
    FWER_FPR, JER_FPR = bootFPR(Dim, nsubj_vec[I], C, X, FWHM, 0):
        