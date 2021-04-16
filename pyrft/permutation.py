"""
Functions to run permutation methods
"""
import sanssouci as ss
import pyrft as pr
import numpy as np
from sklearn.utils import check_random_state
from scipy.stats import t

def boot_contrasts(lat_data, X, C, B = 1000, t_inv = ss.t_inv_linear, replace = True, store_boots = 0):
    """ A function to compute the voxelwise t-statistics for a set of contrasts
      and their (two-sided) p-value by bootstrapping the residuals
  
  Parameters
  -----------------
  lat_data:  a numpy.ndarray of shape (Dim, N) or an object of class field
      giving the data where Dim is the spatial dimension and N is the number of subjects
      if a field then the fibersize must be 1 and the final dimension must be 
      the number of subjects
  X: a numpy.ndarray of shape (N,p)
        giving the covariates (p being the number of parameters)
  C: a numpy.ndarray of shape (L,p)  
        corresponding to the contrast matrix, such that which each row is a 
        contrast vector (where L is the number of constrasts)
  B: int,
      giving the number of bootstraps to do (default is 1000)
  t_inv: a python function 
         that specifies the reference family the default is ss.t_inv_linear which 
         corresponds to the linear reference family form the sansouci package
  replace:  Bool 
      if True (default) then the residuals are sampled with replacement 
      (i.e. a bootstrap), if False then they are sampled without replacement 
      resulting in a permutation of the data
  store_boots: Bool,
          An optional input that allows the bootstrapped p-values to be stored 
          if 1. Default is 0, i.e. no such storage.

  Returns
  -----------------
  minPperm: a numpy.ndarray of shape (1, B),
          where the bth entry (for 1<=b<=B) is the minimum p-value (calculated
          over space) of the bth bootstrap. The 0th entry is the minimum p-value 
          for the original data.
  orig_pvalues: an object of class field,
          giving the p-value (calculated across subjects) of each of the voxels
          and with the same mask as the original data
  pivotal_stats: a numpy.ndarray of shape (1,B)
          whose bth entry is the pivotal statistic of the bth bootstrap,
          i.e. min_{1 <= k <= m} t_k^-1(p_{(k:m)}(T_{n,b}^*)). These quantities 
          are needed for joint error rate control. (At the moment it takes K = m.)
  
  Examples
  -----------------
      
    """
    # Convert the data to be a field if it is not one already
    if type(lat_data) == np.ndarray:
        lat_data = pr.makefield(lat_data)
        
    # Ensure that the fibersize of the field is 1
    if isinstance(lat_data.fibersize, tuple):
        raise Exception("The fibersize of the field must be 1 dimensional")
        
    # Error check the inputs and obtain the size of X
    C, N, p = pr.contrast_error_checking(lat_data,X,C)
        
    ### Main
    # Set random state
    rng = check_random_state(101)
    
    # Initialize the vector of pivotal statistics
    pivotal_stats = np.zeros(B)
    
    # Initialize a vector to store the minimum p-value for each permutation
    minPperm = np.zeros(B)
    
    # Calculate the original statistic (used a the first permutation)
    orig_tstats, residuals = pr.constrast_tstats_noerrorchecking(lat_data, X, C)
    
    # Initialize the p-value field
    orig_pvalues = orig_tstats
    
    # Calculate the p-values 
    # (using abs and multiplying by 2 to obtain the two-sided p-values)
    orig_pvalues.field = 2*(1 - t.cdf(abs(orig_tstats.field), N-p))
    
    # Note need np.ravel as the size of orig_pvalues.field is (Dim, L) i.e. it's not a vector!
    orig_pvalues_sorted = np.array([np.sort(np.ravel(orig_pvalues.field))])
    
    # Get the minimum p-value over voxels and contrasts (include the orignal in the permutation set)
    minPperm[0] = orig_pvalues_sorted[0,0]
    # Obtain the pivotal statistic used for JER control
    pivotal_stats[0] = np.amin(ss.t_inv_linear(orig_pvalues_sorted)) 
    
    # Initialize the boostrap storage!
    bootstore = 0 
    if store_boots:
        L = C.shape[0]
        m = np.prod(lat_data.fieldsize)
        bootstore = np.zeros(L*m, B)
        bootstore[:,0] = orig_pvalues_sorted[0]
        
    # Calculate permuted stats
    # note uses the no error checking version so that the errors are not checked 
    # for each bootstrap!
    lat_data_perm = lat_data
    for b in np.arange(B - 1):
        shuffle_idx = rng.choice(N, N, replace = replace)
        lat_data_perm.field = residuals[...,shuffle_idx]
        permuted_tstats, perm_residuals = pr.constrast_tstats_noerrorchecking(lat_data_perm, X, C)
        
        # Compute the permuted p-values
        # (using abs and multiplying by 2 to obtain the two-sided p-values)
        permuted_pvalues = 2*(1 - t.cdf(abs(permuted_tstats.field), N-p))
        permuted_pvalues = np.array([np.sort(np.ravel(permuted_pvalues))])
        
        #Get the minimum p-value of the permuted data (over voxels and contrasts)
        minPperm[b+1] = permuted_pvalues[0,0]
        
        #Obtain the pivotal statistic - of the permuted data - needed for JER control
        pivotal_stats[b + 1] = np.amin(ss.t_inv_linear(permuted_pvalues)) 
        # could be adjusted for K not m or in general some set A! (i.e. in the step down process)
        
        if store_boots:
            bootstore[:,b+1] = permuted_pvalues[0]
        
    return [minPperm, orig_pvalues, pivotal_stats, bootstore]

def perm_contrasts(lat_data, X, C, B):
    """ A function to compute the voxelwise t-statistics for a set of contrasts
      and their p-value using Manly type permutation
      
  Parameters
  -----------------
  lat_data:  an object of class field consisting of data for N subjects
  X: a numpy.ndarray of shape (N,p)
        giving the covariates (p being the number of parameters)
  C: a numpy.ndarray of shape (L,p)  
        corresponding to the contrast matrix, such that which each row is a 
        contrast vector (where L is the number of constrasts)
  B: int,
      giving the number of bootstraps to do (default is 1000)
  replace:  Bool 
      if True (default) then the residuals are sampled with replacement 
      (i.e. a bootstrap), if False then they are sampled without replacement 
      resulting in a permutation of the data

  Returns
  -----------------
  tstat_field: an object of class field,
          which has spatial size the same as 
                  input data and fibersize equal to the number of contrasts
  
  Examples
  -----------------
      
    """
    # Convert the data to be a field if it is not one already
    if type(lat_data) == np.ndarray:
        lat_data = pr.makefield(lat_data)
    
    # Error check the inputs and obtain the size of X
    C, N, p = pr.contrast_error_checking(lat_data,X,C)
    
    ### Main
    # Set random state
    rng = check_random_state(101)
    
    pivotal_stats = np.zeros((1, B))
    
    # Calculate the original statistic (used a the first permutation)
    orig_tstats = pr.constrast_tstats_noerrorchecking(lat_data, X, C)
    orig_pvalues = orig_tstats
    orig_pvalues.field = 1 - t.cdf(orig_tstats.field, N-p)
    
    # Note need np.ravel as the size of orig_pvalues.field is (Dim, 1) i.e. it's not a vector!
    orig_pvalues_sorted = np.array([np.sort(np.ravel(orig_pvalues.field))])
    pivotal_stats[0,0] = np.amin(ss.t_inv_linear(orig_pvalues_sorted)) 
    
    # Calculate permuted stats
    # note use the no error checking version so that the errors are not checked 
    # for each permutation!
    for b in np.arange(B - 1):
        print(b)
        shuffle_idx = rng.permutation(N)
        permuted_tstats = pr.constrast_tstats_noerrorchecking(lat_data, X[shuffle_idx, :], C)
        permuted_pvalues = 1 - t.cdf(permuted_tstats.field, N-p)
        permuted_pvalues = np.array([np.sort(np.ravel(permuted_pvalues))])
        pivotal_stats[0,b + 1] = np.amin(ss.t_inv_linear(permuted_pvalues)) 
        # could be adjust for K not m or in general some set A! (i.e. in the step down process)
        
    return pivotal_stats, orig_pvalues
    