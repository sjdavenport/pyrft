"""
Functions to run permutation methods
"""
import pyrft as pr
import numpy as np
from sklearn.utils import check_random_state
from scipy.stats import t


def boot_contrasts(lat_data, design, contrast_matrix, n_bootstraps = 1000, template = 'linear', replace = True, store_boots = 0, display_progress = 0):
    """ A function to compute the voxelwise t-statistics for a set of contrasts
      and their (two-sided) p-value by bootstrapping the residuals

    Parameters
    -----------------
    lat_data:  a numpy.ndarray of shape (dim, N) or an object of class field
        giving the data where dim is the spatial dimension and N is the number of subjects
        if a field then the fibersize must be 1 and the final dimension must be
        the number of subjects
    design: a numpy.ndarray of shape (N,p)
        giving the design matrix of covariates (p being the number of parameters)
    contrast_matrix: a numpy.ndarray of shape (L,p)
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
    minp_perm: a numpy.ndarray of shape (1, B),
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
    # 1D
    dim = 5; N = 30; categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
    X = pr.group_design(categ); C = np.array([[1,-1,0],[0,1,-1]]); lat_data = pr.wfield(dim,N)
    minP, orig_pvalues, pivotal_stats, _ = pr.boot_contrasts(lat_data, X, C)

    # 2D
    dim = (10,10); N = 30; categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
    X = pr.group_design(categ); C = np.array([[1,-1,0],[0,1,-1]]); lat_data = pr.wfield(dim,N)
    minP, orig_pvalues, pivotal_stats, _ = pr.boot_contrasts(lat_data, X, C)
    """
    #### Prep
    # Convert the data to be a field if it is not one already
    if isinstance(lat_data, np.ndarray):
        lat_data = pr.make_field(lat_data)

    # Ensure that the fibersize of the field is 1
    if isinstance(lat_data.fibersize, tuple):
        raise Exception("The fibersize of the field must be 1 dimensional")

    # Error check the inputs and obtain the size of X
    contrast_matrix, nsubj, n_params = pr.contrast_error_checking(lat_data,design,contrast_matrix)

    # Obtain the inverse template function (allowing for direct input as well!)
    if isinstance(template, str):
        _, t_inv = pr.t_ref(template)
    else:
        t_inv = template

    #### Main Function
    # Set random state
    rng = check_random_state(101)

    # Initialize the vector of pivotal statistics
    pivotal_stats = np.zeros(n_bootstraps)

    # Initialize a vector to store the minimum p-value for each permutation
    minp_perm = np.zeros(n_bootstraps)

    # Calculate the original statistic (used a the first permutation)
    orig_tstats, residuals = pr.constrast_tstats_noerrorchecking(lat_data, design, contrast_matrix)

    # Initialize the p-value field
    orig_pvalues = orig_tstats

    # Calculate the p-values
    # (using abs and multiplying by 2 to obtain the two-sided p-values)
    orig_pvalues.field = 2*(1 - t.cdf(abs(orig_tstats.field), nsubj-n_params))

    # Note need np.ravel as the size of orig_pvalues.field is (dim, L) i.e. it's not a vector!
    orig_pvalues_sorted = np.array([np.sort(np.ravel(orig_pvalues.field))])

    # Get the minimum p-value over voxels and contrasts (include the orignal in the permutation set)
    minp_perm[0] = orig_pvalues_sorted[0,0]

    # Obtain the pivotal statistic used for JER control
    pivotal_stats[0] = np.amin(t_inv(orig_pvalues_sorted))

    # Initialize the boostrap storage!
    bootstore = 0
    if store_boots:
        # Calculate the number of contrasts
        n_contrasts = contrast_matrix.shape[0]
        masksize_product = np.prod(lat_data.masksize)
        bootstore = np.zeros((n_contrasts*masksize_product, n_bootstraps))
        print(bootstore.shape)
        print(orig_pvalues_sorted.shape)
        bootstore[:,0] = orig_pvalues_sorted[0]

    # Calculate permuted stats
    # note uses the no error checking version so that the errors are not checked
    # for each bootstrap!
    lat_data_perm = lat_data
    for b in np.arange(n_bootstraps - 1):

        # Display progress
        if display_progress:
            pr.modul(b, 1)

        # Obtain a sample with replacement
        shuffle_idx = rng.choice(nsubj, nsubj, replace = replace)
        lat_data_perm.field = residuals[...,shuffle_idx]
        permuted_tstats, _ = pr.constrast_tstats_noerrorchecking(lat_data_perm, design, contrast_matrix)

        # Compute the permuted p-values
        # (using abs and multiplying by 2 to obtain the two-sided p-values)
        permuted_pvalues = 2*(1 - t.cdf(abs(permuted_tstats.field), nsubj-n_params))
        permuted_pvalues = np.array([np.sort(np.ravel(permuted_pvalues))])

        #Get the minimum p-value of the permuted data (over voxels and contrasts)
        minp_perm[b+1] = permuted_pvalues[0,0]

        #Obtain the pivotal statistic - of the permuted data - needed for JER control
        pivotal_stats[b + 1] = np.amin(t_inv(permuted_pvalues))
        # could be adjusted for K not m or in general some set A! (i.e. in the step down process)

        if store_boots:
            bootstore[:,b+1] = permuted_pvalues[0]

    return [minp_perm, orig_pvalues, pivotal_stats, bootstore]

def bootfpr(dim, nsubj, contrast_matrix, fwhm = 0, design = 0, n_bootstraps = 100, niters = 1000, alpha = 0.1, template = 'linear', replace = True, useboot = True):
    """ A function which calculates FWER and JER error rates using niters iterations

  Parameters
  -----------------
  dim: a tuple,
      giving the dimensions of the data to generate
  nsubj: an int,
      giving the number of subjects to use
  C: a numpy.ndarray of shape (L,p)
        corresponding to the contrast matrix, such that which each row is a
        contrast vector (where L is the number of constrasts)
  design: a numpy.ndarray of size (N,p) or an int
        giving the covariates (p being the number of parameters), if set to be
        an integer then random category vectors are generated for each iteration
        and a corresponding design matrix selected
  fwhm: an int,
      giving the fwhm with which to smooth the data (default is 0 i.e. generating
                                                white noise without smoothing)
  B: int,
      giving the number of bootstraps to do (default is 1000)
  niters: int,
      giving the number of iterations to use to estimate the FPR
  alpha: int,
       the alpha level at which to control (default is 0.1)
  t_inv: specifying the reference family (default is the linear reference family)
  replace:  Bool
      if True (default) then the residuals are sampled with replacement
      (i.e. a bootstrap), if False then they are sampled without replacement
      resulting in a permutation of the data
  useboot: Bool,
      determines whether to use bootstrapping to analyse the data or permutation

  Returns
  -----------------
  fpr_fwer: double,
      the false positive rate for FWER control
  fpr_jer: double,
      the false positive rate for JER control

  Examples
  -----------------
# 1D
dim = 5; nsubj = 30; C = np.array([[1,-1,0],[0,1,-1]]);
FWER_FPR, JER_FPR = pr.bootfpr(dim, nsubj, C)

# 2D
dim = (10,10); nsubj = 30; C = np.array([[1,-1,0],[0,1,-1]]);
FWER_FPR, JER_FPR = pr.bootfpr(dim, nsubj, C)
    """
    # Initialize the FPR counter
    n_falsepositives_jer = 0 # jer stands for joint error rate here
    n_falsepositives_fwer = 0

    # Obtain ordered randomness
    rng = check_random_state(101)

    # If the design input is a matrix take this to be the design matrix
    # of the covariates (otherwise a random design is generated - see below)
    if not isinstance(design, int):
        design_2use = design

    # Obtain the inverse template function (allowing for direct input as well!)
    if isinstance(template, str):
        _, t_inv = pr.t_ref(template)
    else:
        # Allow the inverse function to be an input
        t_inv = template

    if len(contrast_matrix.shape) == 1:
        n_contrasts = 1
    else:
        n_contrasts = contrast_matrix.shape[1]

    # Calculate the FPR
    for i in np.arange(niters):
        # Keep track of the progress.
        pr.modul(i,100)

        # Generate the data (i.e. generate stationary random fields)
        lat_data = pr.statnoise(dim,nsubj,fwhm)

        if isinstance(design, int):
            # Generate a random category vector with choices given by the design matrix
            categ = rng.choice(n_contrasts, nsubj, replace = True)

            # Ensure that all categories are present in the category vector
            while len(np.unique(categ)) < n_contrasts:
                print('had rep error')
                categ = rng.choice(n_contrasts, nsubj, replace = True)

            # Generate the corresponding design matrix
            design_2use = pr.group_design(categ)

        if useboot:
            # Implement the bootstrap algorithm on the generated data
            minp_perm, _, pivotal_stats, _ = pr.boot_contrasts(lat_data, design_2use, contrast_matrix, n_bootstraps, t_inv, replace)
        else:
            perm_contrasts(lat_data, design_2use, contrast_matrix, n_bootstraps, t_inv)

        # Calculate the lambda alpha level quantile for JER control
        lambda_quant = np.quantile(pivotal_stats, alpha)

        # Check whether there is a jER false rejection or not
        if pivotal_stats[0] < lambda_quant:
            n_falsepositives_jer = n_falsepositives_jer + 1

        # Calculate the alpha quantile of the permutation distribution of the minimum
        alpha_quantile = np.quantile(minp_perm, alpha)

        if minp_perm[0] < alpha_quantile:
            n_falsepositives_fwer = n_falsepositives_fwer + 1

    # Calculate the false positive rate over all iterations
    fpr_fwer = n_falsepositives_fwer/niters

    # Calculate the standard error
    std_error_fwer = 1.96*np.sqrt(fpr_fwer*(1-fpr_fwer)/niters)

    # Calculate the false positive rate over all iterations
    fpr_jer = n_falsepositives_jer/niters

    # Calculate the standard error
    std_error_jer = 1.96*np.sqrt(fpr_jer*(1-fpr_jer)/niters)

    # Print the results
    print('FWER: ', fpr_fwer, ' +/- ', round(std_error_fwer,4))
    print('JER: ', fpr_jer, ' +/- ', round(std_error_jer,4))

    return fpr_fwer, fpr_jer

def perm_contrasts(lat_data, design, contrast_vector, n_bootstraps = 100, template = 'linear'):
    """ A function to compute the voxelwise t-statistics for a set of contrasts
      and their p-value using Manly type permutation

  Parameters
  -----------------
  lat_data:  an object of class field consisting of data for N subjects
  X: a numpy.ndarray of shape (N,p)
        giving the covariates (p being the number of parameters)
  c: a numpy.ndarray of shape (1,p)
        corresponding to the contrast to use on the data
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
dim = (10,10); N = 30; categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
X = pr.group_design(categ); c = np.array([1,-1,0]); lat_data = pr.wfield(dim,N)
minP, orig_pvalues, pivotal_stats = pr.perm_contrasts(lat_data, X, c)
    """
    # Convert the data to be a field if it is not one already
    if isinstance(lat_data, np.ndarray):
        lat_data = pr.make_field(lat_data)

    # Error check the inputs and obtain the size of X
    contrast_vector, nsubj, n_params = pr.contrast_error_checking(lat_data,design,contrast_vector)

    if contrast_vector.shape[0] > 1:
        raise Exception('c must be a row vector')

      # Obtain the inverse template function (allowing for direct input as well!)
    if isinstance(template, str):
        _, t_inv = pr.t_ref(template)
    else:
        # Allow the inverse function to be an input
        t_inv = template

    ### Main
    # Set random state
    rng = check_random_state(101)

    # Initialize a vector to store the minimum p-value for each permutation
    minp_perm = np.zeros(n_bootstraps)

    # Initialize a vector to store the pivotal statistics for each permutation
    pivotal_stats = np.zeros(n_bootstraps)

    # Calculate the original statistic (used a the first permutation)
    orig_tstats, _ = pr.constrast_tstats_noerrorchecking(lat_data, design, contrast_vector)
    orig_pvalues = orig_tstats
    orig_pvalues.field =  2*(1 - t.cdf(abs(orig_tstats.field), nsubj-n_params))

    # Note need np.ravel as the size of orig_pvalues.field is (dim, 1) i.e. it's not a vector!
    orig_pvalues_sorted = np.array([np.sort(np.ravel(orig_pvalues.field))])

    # Get the minimum p-value over voxels and contrasts (include the orignal in the permutation set)
    minp_perm[0] = orig_pvalues_sorted[0,0]

    # Obtain the pivotal statistics
    pivotal_stats[0] = np.amin(t_inv(orig_pvalues_sorted))

    # Calculate permuted stats
    # note use the no error checking version so that the errors are not checked
    # for each permutation!
    for b in np.arange(n_bootstraps - 1):
        print(b)
        shuffle_idx = rng.permutation(nsubj)
        permuted_tstats, _ = pr.constrast_tstats_noerrorchecking(lat_data, design[shuffle_idx, :], contrast_vector)
        permuted_pvalues = 2*(1 - t.cdf(abs(permuted_tstats.field), nsubj-n_params))
        permuted_pvalues = np.array([np.sort(np.ravel(permuted_pvalues))])

        #Get the minimum p-value of the permuted data (over voxels and contrasts)
        minp_perm[b+1] = permuted_pvalues[0,0]

        # Get the pivotal statistics needed for JER control
        pivotal_stats[b + 1] = np.amin(t_inv(permuted_pvalues))

    return minp_perm, orig_pvalues, pivotal_stats
