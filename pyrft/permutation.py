"""
Functions to run permutation methods
"""
import sanssouci as ss
import pyrft as pr
import numpy as np
from sklearn.utils import check_random_state
from scipy.stats import t

def boot_contrasts(lat_data, X, C, B = 1000, template = 'linear', replace = True, store_boots = 0):
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
Dim = (10,10); N = 30; categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
X = pr.groupX(categ); C = np.array([[1,-1,0],[0,1,-1]]); lat_data = pr.wfield(Dim,N)
minP, orig_pvalues, pivotal_stats, _ = pr.boot_contrasts(lat_data, X, C)
    """
    # Convert the data to be a field if it is not one already
    if type(lat_data) == np.ndarray:
        lat_data = pr.makefield(lat_data)
        
    # Ensure that the fibersize of the field is 1
    if isinstance(lat_data.fibersize, tuple):
        raise Exception("The fibersize of the field must be 1 dimensional")
        
    # Error check the inputs and obtain the size of X
    C, N, p = pr.contrast_error_checking(lat_data,X,C)
    
    # Obtain the inverse template function (allowing for direct input as well!)
    if isinstance(template, str):
        _, t_inv = pr.t_ref(template)
    else:
        t_inv = template
        
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
    pivotal_stats[0] = np.amin(t_inv(orig_pvalues_sorted)) 
    
    # Initialize the boostrap storage!
    bootstore = 0 
    if store_boots:
        L = C.shape[0]
        m = np.prod(lat_data.masksize)
        bootstore = np.zeros((L*m, B))
        print(bootstore.shape)
        print(orig_pvalues_sorted.shape)
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
        pivotal_stats[b + 1] = np.amin(t_inv(permuted_pvalues)) 
        # could be adjusted for K not m or in general some set A! (i.e. in the step down process)
        
        if store_boots:
            bootstore[:,b+1] = permuted_pvalues[0]
        
    return [minPperm, orig_pvalues, pivotal_stats, bootstore]

def bootFPR(Dim, nsubj, C, FWHM = 0,  X = 0, B = 100, niters = 1000, alpha = 0.1, template = 'linear', replace = True, useboot = True):
  """ A function which calculates FWER and JER error rates using niters iterations
    
  Parameters
  -----------------
  Dim: a tuple,
      giving the dimensions of the data to generate
  nsubj: an int,
      giving the number of subjects to use
  C: a numpy.ndarray of shape (L,p)  
        corresponding to the contrast matrix, such that which each row is a 
        contrast vector (where L is the number of constrasts)
  X: a numpy.ndarray of size (N,p) or an int
        giving the covariates (p being the number of parameters), if set to be
        an integer then random category vectors are generated for each iteration
        and a corresponding design matrix selected
  FWHM: an int,
      giving the FWHM with which to smooth the data (default is 0 i.e. generating
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
  FPR_FWER: double,
      the false positive rate for FWER control
  FPR_JER: double,
      the false positive rate for JER control
      
  Examples  
  -----------------
Dim = (10,10); nsubj = 30; C = np.array([[1,-1,0],[0,1,-1]]);
FWER_FPR, JER_FPR = pr.bootFPR(Dim, nsubj, C)
  """
  # Initialize the FPR counter
  nFPs_JER = 0 
  nFPs_FWER = 0 
  
  # Obtain ordered randomness
  rng = check_random_state(101)

  #Set the design matrix to use to be X
  if not isinstance(X, int):
      design_matrix = X
      
  # Obtain the inverse template function (allowing for direct input as well!)
  if isinstance(template, str):
      _, t_inv = pr.t_ref(template)
  else:
      # Allow the inverse function to be an input
      t_inv = template
        
  if len(C.shape) == 1:
      L = 1
  else:
      L = C.shape[1]
      
  # Calculate the FPR
  for I in np.arange(niters):
    # Keep track of the progress.
    pr.modul(I,100)
    
    # Generate the data (i.e. generate stationary random fields) 
    lat_data = pr.statnoise(Dim,nsubj,FWHM)
    
    if isinstance(X, int):
        # Generate a random category vector with choices given by the design matrix
        categ = rng.choice(L, nsubj, replace = True)
        
        # Ensure that all categories are present in the category vector
        while len(np.unique(categ)) < L:
            print('had rep error')
            categ = rng.choice(L, nsubj, replace = True)
            
        # Generate the corresponding design matrix
        design_matrix = pr.groupX(categ)
    
    if useboot:
        # Implement the bootstrap algorithm on the generated data
        minPperm, orig_pvalues, pivotal_stats, _ = pr.boot_contrasts(lat_data, design_matrix, C, B, t_inv, replace)
    else:
        perm_contrasts(lat_data, design_matrix, C, B, t_inv)
    
    # Calculate the lambda alpha level quantile for JER control
    lambda_quant = np.quantile(pivotal_stats, alpha)

    # Check whether there is a jER false rejection or not
    if pivotal_stats[0] < lambda_quant:
       nFPs_JER = nFPs_JER + 1
       
    # Calculate the alpha quantile of the permutation distribution of the minimum
    alpha_quantile = np.quantile(minPperm, alpha)

    if minPperm[0] < alpha_quantile:
        nFPs_FWER = nFPs_FWER + 1
        
  # Calculate the false positive rate over all iterations
  FPR_FWER = nFPs_FWER/niters

  # Calculate the standard error
  std_error_FWER = 1.96*np.sqrt(FPR_FWER*(1-FPR_FWER)/niters)

  # Calculate the false positive rate over all iterations
  FPR_JER = nFPs_JER/niters

  # Calculate the standard error
  std_error_JER = 1.96*np.sqrt(FPR_JER*(1-FPR_JER)/niters)

  # Print the results
  print('FWER: ', FPR_FWER, ' +/- ', round(std_error_FWER,4))
  print('JER: ', FPR_JER, ' +/- ', round(std_error_JER,4))
      
  return FPR_FWER, FPR_JER

def perm_contrasts(lat_data, X, c, B = 100, template = 'linear'):
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
Dim = (10,10); N = 30; categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
X = pr.groupX(categ); c = np.array([1,-1,0]); lat_data = pr.wfield(Dim,N)
minP, orig_pvalues, pivotal_stats = pr.perm_contrasts(lat_data, X, c)
    """
    # Convert the data to be a field if it is not one already
    if type(lat_data) == np.ndarray:
        lat_data = pr.makefield(lat_data)
    
    # Error check the inputs and obtain the size of X
    c, N, p = pr.contrast_error_checking(lat_data,X,c)
    
    if c.shape[0] > 1:
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
    minPperm = np.zeros(B)
    
    # Initialize a vector to store the pivotal statistics for each permutation
    pivotal_stats = np.zeros(B)
    
    # Calculate the original statistic (used a the first permutation)
    orig_tstats, _ = pr.constrast_tstats_noerrorchecking(lat_data, X, c)
    orig_pvalues = orig_tstats
    orig_pvalues.field =  2*(1 - t.cdf(abs(orig_tstats.field), N-p))
   
    # Note need np.ravel as the size of orig_pvalues.field is (Dim, 1) i.e. it's not a vector!
    orig_pvalues_sorted = np.array([np.sort(np.ravel(orig_pvalues.field))])
    
    # Get the minimum p-value over voxels and contrasts (include the orignal in the permutation set)
    minPperm[0] = orig_pvalues_sorted[0,0]
    
    # Obtain the pivotal statistics
    pivotal_stats[0] = np.amin(t_inv(orig_pvalues_sorted)) 
    
    # Calculate permuted stats
    # note use the no error checking version so that the errors are not checked 
    # for each permutation!
    for b in np.arange(B - 1):
        print(b)
        shuffle_idx = rng.permutation(N)
        permuted_tstats, _ = pr.constrast_tstats_noerrorchecking(lat_data, X[shuffle_idx, :], c)
        permuted_pvalues = 2*(1 - t.cdf(abs(permuted_tstats.field), N-p))
        permuted_pvalues = np.array([np.sort(np.ravel(permuted_pvalues))])
        
        #Get the minimum p-value of the permuted data (over voxels and contrasts)
        minPperm[b+1] = permuted_pvalues[0,0]
        
        # Get the pivotal statistics needed for JER control
        pivotal_stats[b + 1] = np.amin(t_inv(permuted_pvalues)) 
        
    return minPperm, orig_pvalues, pivotal_stats