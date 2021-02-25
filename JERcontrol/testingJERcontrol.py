"""
Testing the JER control over contrasts on some examples
"""
import numpy as np
import pyrft as pr
import sanssouci as ss

Dim = (100,100); N = 30; categ = np.random.binomial(1, 0.4, size = N)
X = pr.groupX(categ); C = np.array([[1,-1]]); lat_data = pr.wfield(Dim,N)
lat_data.field = lat_data.field + 1
# Dim = (10,10); N = 30; categ = np.zeros(N)
# X = pr.groupX(categ); C = np.array(1); lat_data = pr.wfield(Dim,N)

B = 1000

m = np.prod(Dim)

pivotal_stats, orig_pvalues = pr.perm_contrasts(lat_data, X, C, B)

# Choose the confidence level
alpha = 0.1

# Obtain the lambda calibration
lambda_quant = np.quantile(pivotal_stats, alpha)

# Gives t_k^L(lambda) = lambda*k/m for k = 1, ..., m
thr = ss.t_linear(lambda_quant, np.arange(1,m+1), m)

# Get the first 10 pvalues
pvals = np.sort(np.ravel(orig_pvalues.field))[:10]

# Compute an upper bound on this
bound = ss.max_fp(pvals, thr)
print(bound)
