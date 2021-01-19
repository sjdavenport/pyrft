# -*- coding: utf-8 -*-
"""
A file contain statistics functions
"""
# Import statements
import numpy as np

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
    out = 
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
  std_dev = data.std(D)
  
  # Compute Cohen's d
  cohensd = xbar/std_dev
  tstat = np.sqrt(nsubj)*cohensd
  
  return(tstat, xbar)
  
  
    