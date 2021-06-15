"""
Functions to run cluster inference
"""
import math
import numpy as np
from skimage import measure
from nilearn.image import get_data, load_img
from nilearn.input_data import NiftiMasker
from nilearn import plotting
import matplotlib.pyplot as plt
from scipy.stats import t
import sanssouci as sa
import pyrft as pr


def find_clusters(test_statistic, cdt, below = bool(0), mask = math.nan, connectivity = 1, two_sample = bool(0), min_cluster_size = 1):
    """ find_clusters
  Parameters
  ---------------------
  test_statistic:   a numpy.nd array,
  cdt:    a double,
        the cluster defining threshold
  below: bool,
      whether to define the clusters above or below the threshold. Default is 0 ie
      clusters above.
  mask
  connectivity
  two_sample
  min_cluster_size

  Returns
  ---------------------
  cluster_image:    a numpy.nd array,
              with the same size as the test-statistic in which the clusters
              above the CDT are labelled each with a different number.

  Examples
  ---------------------
# Clusters above 0.5
cluster_image, cluster_sizes = pr.find_clusters(np.array([[1,0,1],[1,1,0]]), 0.5)
# Clusters below 0.5
cluster_image, cluster_sizes = pr.find_clusters(np.array([[1,0,1],[1,1,0]]), 0.5, below = 1)
# tstat image
f = pr.statnoise((50,50), 20, 10)
tstat, xbar, std_dev = pr.mvtstat(f.field)
cluster_image, c_sizes = pr.find_clusters(tstat, 2)
plt.imshow(cluster_image)
    """

    # Mask the data if that is possible
    if np.sum(np.ravel(mask)) > 0:
        test_statistic = test_statistic*mask

    if two_sample:
        raise Exception("two sample hasn't been implemented yet!")

    if below:
        cluster_image = measure.label((test_statistic < cdt)*(test_statistic > 0), connectivity = connectivity)
    else:
        cluster_image = measure.label(test_statistic > cdt, connectivity = connectivity)

    n_clusters = np.max(cluster_image)
    store_cluster_sizes = np.zeros(1)

    # Use J to keep track of the clusters
    J = 0

    for I in np.arange(n_clusters):
        cluster_index = (cluster_image == (I+1))
        cluster_size = np.sum(cluster_index)
        if cluster_size < min_cluster_size:
            cluster_image[cluster_index] = 0
        else:
            J = J + 1
            store_cluster_sizes = np.append(store_cluster_sizes, cluster_size)
            cluster_image[cluster_index] = J

    # Remove the initial zero
    store_cluster_sizes = store_cluster_sizes[1:]

    return cluster_image, store_cluster_sizes

def cluster_tdp_brain(imgs, design, contrast_matrix, mask, n_bootstraps = 100, fwhm = 4, alpha = 0.1, min_cluster_size = 30, cdt = 0.001):
    """ cluster_tdp_brain calculates the TDP (true discovery proportion) within
    clusters of the test-statistic. This is specifically for brain images
    and enables plotting of these images using the nilearn toolbox
  Parameters
  ---------------------
  imgs
  design
  contrast_matrix
  savedir

  Returns
  ---------------------
  cluster_image:    a numpy.nd array,
              with the same size as the test-statistic in which the clusters
              above the CDT are labelled each with a different number.

  Examples
  ---------------------
    """
    # Obtain the number of parameters in the model
    n_params = contrast_matrix.shape[1]
    
    # Obtain the number of contrasts
    n_contrasts = contrast_matrix.shape[0]

    #Load the data
    masker = NiftiMasker(smoothing_fwhm = fwhm,mask_img = mask, memory='/storage/store2/work/sdavenpo/').fit()
    data = masker.transform(imgs).transpose()

    # Convert the data to a field
    data = pr.makefield(data)


    # Obtain the number of subjects
    nsubj = data.fibersize

    # Obtain the test statistics and convert to p-values
    test_stats, _ = pr.contrast_tstats(data, design, contrast_matrix)
    pvalues = 2*(1 - t.cdf(abs(test_stats.field), nsubj-n_params))

    # Load the mask
    mask = load_img(mask).get_fdata()

    # Obtain a 3D brain image of the p-values for obtaining clusters
    #(squeezing to remove the trailing dimension)
    pvalues_3d = np.squeeze(get_data(masker.inverse_transform(pvalues.transpose())))

    ### Perform Post-hoc inference
    # Run the bootstrapped algorithm
    _, _, pivotal_stats, _ = pr.boot_contrasts(data, design, contrast_matrix, n_bootstraps = n_bootstraps, display_progress = 1)

    # Obtain the lambda calibration
    lambda_quant = np.quantile(pivotal_stats, alpha)

    # Calculate the number of voxels in the mask
    n_vox_in_mask = np.sum(mask[:])

    # Gives t_k^L(lambda) = lambda*k/m for k = 1, ..., m
    thr = sa.t_linear(lambda_quant, np.arange(1,n_vox_in_mask+1), n_vox_in_mask)

    ### Calculate the TDP within each cluster
    if n_contrasts > 1:
        tdp_bounds = np.zeros(pvalues_3d.shape + (n_contrasts,))
    else:
        tdp_bounds = np.zeros(pvalues_3d.shape)

    # Convert the mask to logical
    mask = mask > 0

    # For each cluster calculate the TDP
    for L in np.arange(n_contrasts):
        # Get the clusters of the test-statistic
        cluster_im, cluster_sizes = pr.find_clusters(pvalues_3d[..., L], cdt, below = 1, mask = mask, min_cluster_size = min_cluster_size)

        # Obtain the number of clusters
        n_clusters = len(cluster_sizes)
        
        for I in np.arange(n_clusters):
            # Obtain the logical entries for where each region is
            region_idx = cluster_im[...,L] == (I+1)
        
            # Compute the TP bound
            bound = sa.max_fp(pvalues_3d[region_idx], thr)
            tdp_bounds[region_idx, L] = (np.sum(region_idx) - bound)/np.sum(region_idx)

    return tdp_bounds