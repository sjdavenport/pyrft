"""
Testing bootstrapping in the linear model
"""
import pyrft as pr
import numpy as np
Dim = 1; N = 30; categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
X = pr.groupX(categ); C = np.array([[1,-1,0],[0,1,-1]]); lat_data = pr.wfield(Dim,N)

minP, orig_pvalues, pivotal_stats = pr.boot_contrasts(lat_data, X, C)

# %%
### One sample, one voxel test
alpha = 0.1; niters = 1000;
Dim = (10,10); N = 20; categ = np.zeros(N)
X = pr.groupX(categ); C = np.array(1); 

number_of_false_positives = 0
store_origs = np.zeros((1,niters))
for I in np.arange(niters):
    print(I) 
    lat_data = pr.wfield(Dim,N)
    minPperm, orig_pvalues, pivotal_stats = pr.boot_contrasts(lat_data, X, C, 100)
    alpha_quantile = np.quantile(minPperm, alpha)
    store_origs[0,I] = minPperm[0]
    if minPperm[0] < alpha_quantile:
        number_of_false_positives = number_of_false_positives + 1
        
FPR = number_of_false_positives/niters

# %% Multiple contrasts - global null is true
alpha = 0.1; niters = 1000;
Dim = (10,10); N = 30; 
categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
X = pr.groupX(categ); 
C = np.array([[1,-1,0],[0,1,-1]]); 
B = 100

number_of_false_positives = 0
store_origs = np.zeros((1,niters))
for I in np.arange(niters):
    print(I) 
    lat_data = pr.wfield(Dim,N)
    minPperm, orig_pvalues, pivotal_stats = pr.boot_contrasts(lat_data, X, C, B)
    alpha_quantile = np.quantile(minPperm, alpha)
    store_origs[0,I] = minPperm[0]
    if minPperm[0] < alpha_quantile:
        number_of_false_positives = number_of_false_positives + 1
        
FPR = number_of_false_positives/niters

# %% Multiple contrasts - global null is false
alpha = 0.1; niters = 1000;
Dim = (10,10); N = 30; 
from sklearn.utils import check_random_state
rng = check_random_state(101)
categ = rng.choice(3, N, replace = True)
X = pr.groupX(categ); 
C = np.array([1,-1,0]); 
B = 100

number_of_false_positives = 0
store_origs = np.zeros((1,niters))
for I in np.arange(niters):
    print(I)
    lat_data = pr.wfield(Dim,N)
    minPperm, orig_pvalues, pivotal_stats = pr.boot_contrasts(lat_data, X, C, B)
    alpha_quantile = np.quantile(minPperm, alpha)
    store_origs[0,I] = minPperm[0]
    if minPperm[0] < alpha_quantile:
        number_of_false_positives = number_of_false_positives + 1
        
FPR = number_of_false_positives/niters
 
# Note that this give inflated false positives for low N! E.g. N = 30! This gets better
# as N is increased but worse and worse as Dim increases so Anderson may have missed
# it in his FL paper as I'm fairly sure that the tests there were only done in 1D!!

# %% Multiple contrasts - testing strong control
alpha = 0.1; niters = 1000;
Dim = (10,10); N = 100; 
from sklearn.utils import check_random_state
rng = check_random_state(101)
categ = rng.choice(3, N, replace = True)
X = pr.groupX(categ); 
C = np.array([1,-1,0]); 
w2 = np.where(categ==2)
B = 100
signal = 4;

number_of_false_positives = 0
store_origs = np.zeros((1,niters))
for I in np.arange(niters):
    print(I) 
    lat_data = pr.wfield(Dim,N)
    lat_data.field[:,:,w2]=  lat_data.field[:,:,w2] + signal
    minPperm, orig_pvalues, pivotal_stats = pr.boot_contrasts(lat_data, X, C, B)
    alpha_quantile = np.quantile(minPperm, alpha)
    store_origs[0,I] = minPperm[0]
    if minPperm[0] < alpha_quantile:
        number_of_false_positives = number_of_false_positives + 1
        
FPR = number_of_false_positives/niters

