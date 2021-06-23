"""
Functions to control the fdr
"""
import numpy as np

def fdr_bh( pvalues, alpha = 0.05 ):
    """ fdr_bh( pvalues, alpha ) implements the Benjamini-Hochberg procedure on 
  a numpy array of pvalues, controlling the FDR to a level alpha

  Parameters
  ---------------------
  pvalues       a vector of p-values
  alpha         the significance level, default is 0.05

  Returns
  ---------------------
  rejection_ind: a boolean numpy array,
      with the same size as pvalues such that a
              given entry is 1 if that point is rejected and 0 otherwise
  n_rejections: int,
      the total number of rejections
  rejection_locs:  a int numpy array
      the locations of the rejections

  Examples
  ---------------------
from scipy.stats import norm
nvals = 100; normal_rvs = np.random.randn(1,100)[0]
normal_rvs[0:20] = normal_rvs[0:20] + 2
pvalues = 1 - norm.cdf(normal_rvs)
rejection_ind, n_rejections, sig_locs = fdr_bh(pvalues)
print(sig_locs)
    """
    # Get the dimension of the pvalues
    dim = pvalues.shape
    
    # Need the sort index for python!
    pvalues_vector = np.ravel(pvalues);
    sort_index = np.argsort(pvalues_vector)
    sorted_pvalues = pvalues_vector[sort_index]
    n_pvals = len(pvalues_vector)

    bh_upper = (np.arange(n_pvals) + 1)*alpha/n_pvals
    
    bh_vector = sorted_pvalues <= bh_upper
    
    # Find the indices j for which p_(j) \leq alpha*j/npvals 
    # Note need to add + 1 to account for python indexing
    find_lessthans = np.where(bh_vector)[0] + 1

    if find_lessthans.shape[0] == 0:
        # If there are no rejections
        n_rejections = 0
        rejection_locs = np.zeros(0)
        rejection_ind = np.zeros(dim)
    else:
        # If there are rejections, find and store them and return them as output
        
        # Calculate the number of rejections
        n_rejections = find_lessthans[-1]
        
        # Obtain the rejections location indices
        rejection_locs = np.sort(sort_index[0:n_rejections])
        
        # Initialize a boolean vector of location of rejections
        rejection_ind = np.zeros(n_pvals, dtype = 'bool')
        
        # Set each rejection to 1
        rejection_ind[rejection_locs] = 1

        # Reshape rejection_ind so that it has the same size as pvalues
        rejection_ind = rejection_ind.reshape(dim)
    
    return rejection_ind, n_rejections, rejection_locs 