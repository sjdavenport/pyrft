"""
Additional functions for the sansaouci toolbox
"""
import sanssouci as sa
from scipy.stats import beta
import numpy as np
import pyrft as pr

def t_beta(lamb, k, n_hypotheses):
    """ A function to compute the template for the beta family

    Parameters
    -----------------
    lamb   double,
          the lambda value to evaluate at
    k: int,
          indexes the reference family
    m: int,
          the total number of p-values

    Returns
    -----------------
    a numpy.ndarray of shape ()

    Examples
    -----------------
    lamb = 0.1
    pr.t_beta(lamb, 1, 5)

    # Plot the beta curves
    import matplotlib.pyplot as plt
    i = 1000;
    lamb_vec = np.arange(i)/i
    plt.plot(lamb_vec, pr.t_beta(lamb_vec, 1, 10))

    lamb = 0.9; m = 1000; k = np.arange(m)
    plt.plot(k, pr.t_beta(lamb, k, m))

    lamb = 0.001; m = 1000; k = np.arange(100)
    plt.plot(k, pr.t_beta(lamb, k, m))

    lamb = np.exp(-10); m = 50; k = np.arange(m)
    plt.plot(k, pr.t_beta(lamb, k, m))
    """
    # t_k^B(lambda) = F^{-1}(lambda) where F (beta.ppf) is the cdf of the
    # beta(k, m+1-k) distribution. This yields the lambda quantile of this distribution
    return beta.ppf(lamb, k, n_hypotheses+1-k)

def t_inv_beta(set_of_pvalues):
    """ A function to compute the inverse template for the beta family

    Parameters
    -----------------
    p0: a numpy.ndarray of shape (B,m) ,
          where m is the number of null hypotheses and B is typically the number
          of permutations/bootstraps that contains the values on which to apply (column wise)
          the inverse beta reference family

    Returns
    -----------------
    a numpy.ndarray of shape ()

    Examples
    data = np.random.uniform(0,1,(10,10))
    pr.t_inv_beta(data)
    -----------------
    """
    # Obtain the number of null hypotheses
    n_hypotheses = set_of_pvalues.shape[1]

    # Initialize the matrix of transformed p-values
    transformed_pvalues = np.zeros(set_of_pvalues.shape)

    # Transformed each column via the beta pdf
    # (t_k^B)^{-1}(lambda) = F(lambda) where F (beta.pdf) is the cdf of the
    # beta(k, m+1-k) distribution.
    for k in np.arange(n_hypotheses):
        transformed_pvalues[:,k] = beta.cdf(set_of_pvalues[:,k], k+1, n_hypotheses+1-(k+1))

    return transformed_pvalues

def t_ref(template = 'linear'):
    """ A function to compute the inverse template for the beta family

    Parameters
    -----------------
    template: str,
          a string specifying the template to use, the options are 'linear' (default)
          and 'beta'

    Returns
    -----------------
    t_func:  function,

    t_inv: function,


    Examples
    -----------------
    % Obtaining the linear template functions
    t_func, t_inv = t_ref()

    % Obtaining the beta template functions
    t_func, t_inv = t_ref('beta')
    """
    if template == 'linear' or template == 'simes':
        t_func = pr.t_linear
        t_inv = pr.t_inv_linear
    elif template == 'beta' or template == 'b':
        t_func = pr.t_beta
        t_inv = pr.t_inv_beta
    else:
        raise Exception('The specified template is not available or has been incorrectly input')

    return t_func, t_inv

def  t_linear(alpha, k, m):
  """ 
  A function to compute the linear template
  ----------------------------------------------------------------------------
  Inputs:
  - alpha: a significance level 0 < alpha < 1
  - k: an integer between 1 and K the total number of reference families?
  - m: the total number of hypotheses being tested

  Returns:
  - out:    a numpy array with values 
  ----------------------------------------------------------------------------
  Examples:
    alpha = 0.05; k = 
    out = t_linear(p0)
  ----------------------------------------------------------------------------
  """
  return alpha * k / (m*1.0)


def  t_inv_linear(p0):
  """ QQ: Why is it of this form?
  A function to compute t_k^(-1)
  ----------------------------------------------------------------------------
  ARGUMENTS
  - p0: an array of size B by m (B: nperm, m: number of null hypotheses)
  ----------------------------------------------------------------------------
  OUTPUT
  - out:    a numpy array such that out_{bn} = p0_{bn}*m/b
  ----------------------------------------------------------------------------
  EXAMPLES
    out = t_inv_linear(p0)
  ----------------------------------------------------------------------------
  Warning: In case the last command does not work, you can use alternatively:

  out=pval0_sorted_all*1.
  for i in range(pval0_sorted_all.shape[0]):
    out[i,:]=pval0_sorted_all[i,:]/normalized_ranks[:]
  return out
  """
  
  # Obtain the number of columns of p0: the total number of hypotheses being tested 
  m = p0.shape[1]

  # Generate a vector: (1/m,2/m,...,1)
  normalized_ranks = (np.arange(m)+1)/float(m)
  return p0/normalized_ranks