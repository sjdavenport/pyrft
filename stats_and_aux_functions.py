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
  - C:         an L by p numpy matrix for which each row is a contrast (where 
               L is the number of constrasts)
  ----------------------------------------------------------------------------
  OUTPUT:
  - tstat_field   an object of class field which has spatial size the same as 
                  input data and fibersize equal to the number of contrasts
  ----------------------------------------------------------------------------
  EXAMPLES:
  # One Sample tstat
  Dim = (3,3); N = 30; categ = np.zeros(N)
  X = groupX(categ); C = np.array(1); lat_data = pr.wfield(Dim,N)
  tstat = contrast_tstats(lat_data, X, C)  
  # Compare to mvtstat:
  print(tstat.field.reshape(lat_data.masksize)); print(mvtstat(lat_data.field)[0])
      
  # Two Sample tstat
  Dim = (10,10); N = 30; categ = np.random.binomial(1, 0.4, size = N)
  X = groupX(categ); C = np.array((1,-1)); lat_data = pr.wfield(Dim,N)
  tstats = contrast_tstats(lat_data, X, C)
  
  # 3 Sample tstat (lol)
  Dim = (10,10); N = 30; categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
  X = groupX(categ); C = np.array([[1,-1,0],[0,1,-1]]); lat_data = pr.wfield(Dim,N)
  tstats = contrast_tstats(lat_data, X, C)
  ----------------------------------------------------------------------------
    """
      # Ensure C is a numpy array
    if type(C) != np.ndarray:
        raise Exception("C must be a numpy array")
        
    # Ensure that C is a numpy matrix
    if len(C.shape) == 0:
        C = np.array([[C]])
    elif len(C.shape) == 1:
        C = np.array([C])
    elif len(C.shape) > 2:
        raise Exception("C must be a matrix not a larger array")
        
    # Calculate the number of parameters in C
    C_p = C.shape[1] # parameters
    
    # Calculate the number of parameters p and subjects N
    N = X.shape[0] # subjects
    p = X.shape[1] # parameters
     
    # Ensure that the dimensions of X and C are compatible
    if p != C_p:
        raise Exception('The dimensions of X and C do not match')
      
    # Ensure that the dimensions of X and lat_data are compatible
    if N != lat_data.fibersize:
        raise Exception('The number of subjects in of X and lat_data do not match')
    
    # Having now run error checking calculate the contrast t-statistics
    tstat_field = constrast_tstats_noerrorchecking(lat_data, X, C)
    
    return tstat_field

def constrast_tstats_noerrorchecking(lat_data, X, C):
    """ A function to compute the voxelwise t-statistics for a set of contrasts
    but with no error checking! For input into permutation so you do not have to
    run the error checking every time.
  ----------------------------------------------------------------------------
  ARGUMENTS:
  - lat_data:  an object of class field consisting of data for N subjects
  - X:         an N by p numpy array of covariates (p being the number of parameters)
  - C:         an L by p numpy matrix for which each row is a contrast (where 
               L is the number of constrasts)
  ----------------------------------------------------------------------------
  OUTPUT:
  - tstat_field   an object of class field which has spatial size the same as 
                  input data and fibersize equal to the number of contrasts
  ----------------------------------------------------------------------------
  EXAMPLES:
  # One Sample tstat
  Dim = (3,3); N = 30; categ = np.zeros(N)
  X = groupX(categ); C = np.array([[1]]); lat_data = pr.wfield(Dim,N)
  tstat = _constrast_tstats_4perm(lat_data, X, C)  
  # Compare to mvtstat:
  print(tstat.field.reshape(lat_data.masksize)); print(mvtstat(lat_data.field)[0])
      
  # Two Sample tstat
  Dim = (10,10); N = 30; categ = np.random.binomial(1, 0.4, size = N)
  X = groupX(categ); C = np.array([[1,-1]]); lat_data = pr.wfield(Dim,N)
  tstats = _constrast_tstats_4perm(lat_data, X, C)
  
  # 3 Sample tstat (lol)
  Dim = (10,10); N = 30; categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
  X = groupX(categ); C = np.array([[1,-1,0],[0,1,-1]]); lat_data = pr.wfield(Dim,N)
  tstats = _constrast_tstats_4perm(lat_data, X, C)
  ----------------------------------------------------------------------------
    """
    # Calculate the number of contrasts
    L = C.shape[0]  # constrasts
    
    # Calculate the number of parameters p and subjects N
    N = X.shape[0] # subjects
    p = X.shape[1] # parameters
     
    #rfmate = np.identity(p) - np.dot(X, np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)))
    # Calculate (X^TX)^(-1)
    XTXinv = np.linalg.inv(X.T @ X) 
    
    # Calculate betahat (note leave the extra shaped 1 in as will be remultipling
    # with the contrast vectors!)
    betahat = XTXinv @ X.T @ lat_data.field.reshape( lat_data.fieldsize + (1,) ) 
    
    # Calculate the residual forming matrix
    rfmate = np.identity(N) - X @ XTXinv @ X.T
    
    # Compute the estimate of the variance via the residuals (I-P)Y
    # Uses a trick adding (1,) so that multiplication is along the last column!
    # Note no need to reshape back not doing so will be useful when dividing by the std
    residuals = rfmate @ lat_data.field.reshape( lat_data.fieldsize + (1,) ) 
    
    # Square and sum over subjects to calculate the variance
    # This assumes that X has rank p!
    std_est = (np.sum(residuals**2,lat_data.D)/(N-p))**(1/2)
    
    # Compute the t-statistics
    tstats = (C @ betahat).reshape(lat_data.masksize + (L,))/std_est
    
    # Scale by the scaling constants to ensure var 1
    for l in np.arange(L):
        scaling_constant = (C[l,:] @ XTXinv @ C[l,:])**(1/2)
        tstats[...,l] = tstats[...,l]/scaling_constant
        
    # Generate the field of tstats
    tstat_field = pr.Field(tstats, lat_data.mask)
    
    return tstat_field

def groupX(categ):
  """ A function to compute the covariate matrix X for a given set of categories
  ----------------------------------------------------------------------------
  ARGUMENTS:
  - categ:  a tuple of integers of length N(number of subjects) where each entry is 
            number of the category that a given subject belongs to 
            (enumerated from 0 to ncateg - 1)
            E.g: (0,1,1,0) corresponds to 4 subjects, 2 categories and
                 (0,1,2,3,3,2) corresponds to 6 subjects and 4 categories
            Could make a category class!
  ----------------------------------------------------------------------------
  OUTPUT:
  - X:      a design matrix that can be used to assign the correct categories
  ----------------------------------------------------------------------------
  EXAMPLES:
  categ = (0,1,1,0); groupX(categ)
  ----------------------------------------------------------------------------
  """
  # Calculate the number of subjects
  N = len(categ) 
  
  # Calculate the number of parameters i.e. the number of distinct groups
  p = len(np.unique(categ))
  
  # Ensure that the number of categories is not too high!
  if np.max(categ) > p - 1:
      raise Exception("the maximum category number should not exceed one minus the number of categories")
  
  # Initialize the design matrix
  X = np.zeros((N,p))
  
  # Set the elements of the design matrix by assigning each subject a category
  for I in np.arange(N):
      X[I, int(categ[I])] = 1 # change so you do this all at once if possible!
      
  return X
  
    