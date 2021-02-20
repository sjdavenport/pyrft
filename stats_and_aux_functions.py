# -*- coding: utf-8 -*-
"""
A file contain statistics functions
"""
# Import statements
import numpy as np
import sys
sys.path.insert(0, 'C:\\Users\\12SDa\\davenpor\\davenpor\\Toolboxes' )
import pyrft as pr


def mvtstat(data):
  """ A function to compute the multivariate t-statistic
  ----------------------------------------------------------------------------
  Inputs:
  - data:   a Dim by nsubj numpy array

  Returns:
  - tstat:  a numpy array of size Dim where each entry is the t-statistic
  - mean:   a numpy array of size Dim where each entry is the mean
  - std:    a numpy array of size Dim where each entry is the standard deviation
  ----------------------------------------------------------------------------
  Examples:
  # tstat of random noise
  noise = np.random.randn(50,50,20); arrays = mvtstat(noise);  tstat = arrays[0]
  # For comparison to MATLAB
  a = np.arange(12).reshape((3,4)).transpose()+1; tstat = mvtstat(a)[0]
  ----------------------------------------------------------------------------
  """
  
  # Obtain the size of the array
  sD = np.shape(data)
  
  # Obtain the dimensions
  Dim = sD[0:-1]
  
  # Obtain the number of dimensions
  D = len(Dim)
  
  # Obtain the number of subjects
  nsubj = sD[-1]
  
  # Get the mean and stanard deviation along the number of subjects
  xbar = data.mean(D) # Remember in python D is the last dimension of a D+1 array
  
  # Calculate the standard deviation (multiplying to ensure the population std is used!)
  std_dev = data.std(D)*np.sqrt((nsubj/(nsubj-1.)))
  
  # Compute Cohen's d
  cohensd = xbar/std_dev
  tstat = np.sqrt(nsubj)*cohensd
  
  return(tstat, xbar, std_dev)

def contrast_tstats(lat_data, X, C):
  """ A function to compute the voxelwise t-statistics for a set of contrasts
  ----------------------------------------------------------------------------
  ARGUMENTS:
  - lat_data:  an object of class field consisting of data for N subjects
  - X:         an N by p numpy array of covariates (p being the number of parameters)
  - C:         a matrix for which each row is a contrast
  ----------------------------------------------------------------------------
  OUTPUT:
  - tstat_field   an object of class field which has spatial size the same as 
                  input data and fibersize equal to the number of contrasts
  ----------------------------------------------------------------------------
  EXAMPLES:
      
  ----------------------------------------------------------------------------
  """
  #Need to check that the dimensions of X and C and lat_data match!
  L = C.shape[1] # Calculate the number of contrasts
  if X.shape[0] != L:
      raise Exception('The dimensions of X and C don''t match')
  
  # Calculate the number of parameters p and subjects N
  N = X.shape[0]
  p = X.shape[1]
  
  #rfmate = np.identity(p) - np.dot(X, np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)))
  # Calculate (X^TX)^(-1)
  XTXinv = np.linalg.inv(np.transpose(X) @ X) 
  
  # Calculate betahat (note leave the extra shaped 1 in as will be remultipling
  # with the contrast vectors!)
  betahat = XTXinv @ np.transpose(X) @ lat_data.data.reshape( lat_data.fieldsize + (1,) ) 
  
  # Calculate the residual forming matrix
  rfmate = np.identity(p) - X @ XTXinv @ np.transpose(X)
  
  # Compute the estimate of the variance via the residuals (I-P)Y
  # Uses a trick adding (1,) so that multiplication is along the last column!
  # Note no need to reshape back not doing so will be useful when dividing by the std
  residuals = rfmate @ lat_data.data.reshape( lat_data.fieldsize + (1,) ) 
  
  # Square and sum over subjects to calculate the variance
  # This assumes that X has rank p!
  std_est = (np.sum(residuals**2,lat_data.D)/(N-p))**(1/2)
  
  # For each row compute the test-statistic 
  tstat_field.data = pr.Field((C @ betahat)/std_est, lat_data.mask)
      
  return tstat_field
  
    