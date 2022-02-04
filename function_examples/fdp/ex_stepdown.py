"""
Testing the stepdown function
"""
import pyrft as pr
import numpy as np
from sklearn.utils import check_random_state

dim = (10,10)
nsubj = 100
fwhm = 4
lat_data = pr.statnoise(dim,nsubj,fwhm)

contrast_matrix = np.array([[1,-1,0],[0,1,-1]])
n_groups = contrast_matrix.shape[1]
rng = check_random_state(101)
categ = rng.choice(n_groups, nsubj, replace = True)
design_2use = pr.group_design(categ)
pi0 = 0.5
lat_data, signal = pr.random_signal_locations(lat_data, categ, contrast_matrix, pi0)
minp_perm, orig_pvalues, pivotal_stats, bootstore = pr.boot_contrasts(lat_data, design_2use, contrast_matrix, store_boots = 1)

# %%
plt.imshow(orig_pvalues.field[:,:,1])

# %%
alpha = 0.1
lambda_quant_orig = np.quantile(pivotal_stats, alpha)
lambda_quant_sd, stepdownset = pr.step_down(bootstore, alpha)

