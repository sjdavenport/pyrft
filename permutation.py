"""
Functions to run permutation methods
"""
import sys
sys.path.insert(0, 'C:\\Users\\12SDa\\global\\Intern\\SanSouciCode')
import sanssouci as ss
import numpy as np
sys.path.insert(0, 'C:\\Users\\12SDa\\davenpor\\davenpor\\Toolboxes' )
import pyrft as pr
from sklearn.utils import check_random_state
from scipy.stats import t

def perm_contrasts(lat_data, X, C, B):
    """ A function to compute the voxelwise t-statistics for a set of contrasts
      and their p-value using permutation
  ----------------------------------------------------------------------------
  ARGUMENTS:
  - lat_data:  an object of class field consisting of data for N subjects
  - X:         an N by p numpy array of covariates (p being the number of parameters)
  - C:         an L by p numpy matrix for which each row is a contrast (where 
               L is the number of constrasts)
  - B:         
  ----------------------------------------------------------------------------
  OUTPUT:
  - tstat_field   an object of class field which has spatial size the same as 
                  input data and fibersize equal to the number of contrasts
  ----------------------------------------------------------------------------
  EXAMPLES:
      
  ----------------------------------------------------------------------------
    """
    
    ### Error Checking
    # Ensure that C is a numpy array 
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
    
    ### Main
    # Set random state
    rng = check_random_state(101)
    
    pivotal_stats = np.zeros((1, B))
    
    # Calculate the original statistic (used a the first permutation)
    orig_tstats = pr.constrast_tstats_noerrorchecking(lat_data, X, C)
    orig_pvalues = orig_tstats
    orig_pvalues.field = 1 - t.cdf(orig_tstats.field, N-p)
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
    