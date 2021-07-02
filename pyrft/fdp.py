"""
Functions to control the fdr
"""
import numpy as np
from scipy.ndimage import label
from scipy.stats import norm

from nilearn.input_data import NiftiMasker
from nilearn.image import get_data, math_img
import pyrft as pr

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

def ari(pvalues, subset, alpha= 0.1 ):
    """ ari implements all resolution inference (Rosenblatt 2018, see below) and 
    calulates bounds on the TDP for p-values within the specified subset

    Parameters
    ----------
    pvalues:  a numpy.nd array,
        containing the p-values

    subset:   numpy boolean array,
        the same size as pvalues indicating which elements
        to use in the subset or 

    alpha: float or list, optional
        level of control on the true positive rate, aka true dsicovery
        proportion

    Returns
    -------
    tdp:  float,
        the true discovery proportion found by applying ARI

    Note
    ----
    This implements the method described in:

    Rosenblatt JD, Finos L, Weeda WD, Solari A, Goeman JJ. All-Resolutions
    Inference for brain imaging. Neuroimage. 2018 Nov 1;181:786-796. doi:
    10.1016/j.neuroimage.2018.07.060
    """

    hommel_value = _compute_hommel_value(pvalues, alpha)
    subset_pvalues = pvalues[subset]
    tdp = _true_positive_fraction(subset_pvalues, hommel_value, alpha)
    
    return tdp

def _compute_hommel_value(pvalues, alpha = 0.1):
    """ This function computes the hommel value needed for all resolution 
    inference

  Parameters
  ---------------------
  pvalues:      a numpy.nd array,
      containing the pvalues
  alpha:        float,    
      the significance level, default is 0.1

  Returns
  ---------------------
  hommel_value: the value from implementing the hommel procedure

  Examples
  ---------------------
zvals = np.random.randn(1, 100)
zvals = zvals[0]
pvals = norm.sf(zvals)
out =  _compute_hommel_value(pvals, alpha = 0.1)
    """
    # Ensure that the alpha level is between 0 and 1
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha should be between 0 and 1')
        
    # Sort the pvalues
    pvalues = np.sort(np.ravel(pvalues))
    
    # Calculate the number of pvalues
    n_pvals = len(pvalues)

    if len(pvalues) == 1:
        return pvalues[0] > alpha
    if pvalues[0] > alpha:
        return n_pvals
    
    # Discuss with Bertrand what these lines do
    slopes = (alpha - pvalues[: - 1]) / np.arange(n_pvals, 1, -1)
    slope = np.max(slopes)
    hommel_value = np.trunc(n_pvals + (alpha - slope * n_pvals) / slope)
    
    return np.minimum(hommel_value, n_pvals)

def _true_positive_fraction(pvalues, hommel_value, alpha):
    """Given a bunch of pvalues return the true positive fraction

    Parameters
    ----------
    pvalues: array,
            a set of pvalues from which the FDR is computed
    hommel_value: int,
           the Hommel value, used in the computations
    alpha: float,
           desired FDR control

    Returns
    -------
    threshold: float,
               Estimated true positive fraction in the set of values
    """
    # Sort the pvalues
    pvalues = np.sort(np.ravel(pvalues))
    
    # Calculate the number of pvalues
    n_pvals = len(pvalues)
    
    c = np.ceil((hommel_value * pvalues) / alpha)
    unique_c, counts = np.unique(c, return_counts=True)
    criterion = 1 - unique_c + np.cumsum(counts)
    proportion_true_discoveries = np.maximum(0, criterion.max() / n_pvals)
    
    return proportion_true_discoveries
