# -*- coding: utf-8 -*-
"""
Testing the functions in permutation.py
"""
import numpy as np
import pyrft as pr

def test_boot_contrasts():
    np.random.seed(10)
    for I in np.arange(2):
        if I == 1:
            # 1D example
            dim = 5
        else:
            # 2D example
            dim = (10,10)

    N = 30
    categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
    X = pr.group_design(categ)
    C = np.array([[1,-1,0],[0,1,-1]])
    lat_data = pr.wfield(dim,N)
    B = 100;
    minp, orig_pvalues, pivotal_stats, boot_stores = pr.boot_contrasts(lat_data, X, C, B, store_boots = 1)

    # Testing minp
    assert minp.min() > 0
    assert minp.max() < 1
    assert minp.shape == (B,)

    # Testing orig_pvalues
    assert isinstance(orig_pvalues, pr.classes.Field)
    assert orig_pvalues.D == 1
    assert orig_pvalues.fieldsize == (5,2)
    assert orig_pvalues.masksize == (1,5)

    # Testing pivotal_stats
    assert pivotal_stats.shape == (B,)
    assert pivotal_stats.min() > 0
    assert pivotal_stats.max() < 1

    # Testing bootstrap storage
    assert boot_stores.shape == (10,B)
    assert boot_stores.min() > 0
    assert boot_stores.max() < 1

# boot_fpr takes too long to run in order to test it

def test_perm_contrasts():
    np.random.seed(10)
    for I in np.arange(2):
        if I == 1:
            # 1D example
            dim = 5
        else:
            # 2D example
            dim = (10,10)

    N = 30; categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
    X = pr.group_design(categ)
    c = np.array([1,-1,0])
    lat_data = pr.wfield(dim,N)
    B = 100;
    minp, orig_pvalues, pivotal_stats = pr.perm_contrasts(lat_data, X, c, B)

    # Testing minp
    assert minp.min() > 0
    assert minp.max() < 1
    assert minp.shape == (B,)

    # Testing orig_pvalues
    assert isinstance(orig_pvalues, pr.classes.Field)
    assert orig_pvalues.D == 1
    assert orig_pvalues.fieldsize == (5,1)
    assert orig_pvalues.masksize == (1,5)

    # Testing pivotal_stats
    assert pivotal_stats.shape == (B,)
    assert pivotal_stats.min() > 0
    assert pivotal_stats.max() < 1