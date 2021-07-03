# -*- coding: utf-8 -*-
"""
Testing File
"""
# Import necessary modules
import numpy as np
import pyrft as pr
import os

# The location where the pyrft module is saved (need to specify yourself)
pyrft_loc = 'C:/Users/12SDa/davenpor/davenpor/Toolboxes/pyrft/'

# Where data is stored
data_store = pyrft_loc + 'tests2/test_data/'

def test_boot_contrasts():
    np.random.seed(10)
    Dim = 5; N = 30; categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
    X = pr.groupX(categ); C = np.array([[1,-1,0],[0,1,-1]]); lat_data = pr.wfield(Dim,N)
    minP, orig_pvalues, pivotal_stats, _ = pr.boot_contrasts(lat_data, X, C)
    
    file_name = 'test_boot_contrasts_data.npz'
    file_loc = data_store + file_name
    if not os.path.exists(file_loc):
       np.savez(file_loc, minP = minP, orig_pvalues = orig_pvalues.field, pivotal_stats = pivotal_stats)
    else:
       results = np.load(file_loc, allow_pickle = True)
       assert((minP == results['minP']).all())
       assert((orig_pvalues.field == results['orig_pvalues']).all())
       assert((pivotal_stats == results['pivotal_stats']).all())

def test_perm_contrasts():
    np.random.seed(10)
    Dim = (2,2); N = 30; categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
    X = pr.groupX(categ); c = np.array([1,-1,0]); lat_data = pr.wfield(Dim,N)
    minP, orig_pvalues, pivotal_stats = pr.perm_contrasts(lat_data, X, c)
    
    file_name = 'test_perm_contrasts_data.npz'
    file_loc = data_store + file_name
    if not os.path.exists(file_loc):
       np.savez(file_loc, minP = minP, orig_pvalues = orig_pvalues.field, pivotal_stats = pivotal_stats)
    else:
       print('Running main')
       results = np.load(file_loc, allow_pickle = True)
       assert((minP == results['minP']).all())
       assert((orig_pvalues.field == results['orig_pvalues']).all())
       assert((pivotal_stats == results['pivotal_stats']).all())
