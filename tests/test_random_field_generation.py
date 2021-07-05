# -*- coding: utf-8 -*-
"""
Testing functions for the random field generation functions
"""
import numpy as np
import pyrft as pr

def test_smooth():
    nvox = 50
    for I in np.arange(2):
        dim = tuple(np.tile(nvox,I + 2))
        D = len(dim)
        nsubj = 10
        fwhm = 8
        f = pr.wfield(dim, 10)
        smooth_f = pr.smooth(f, fwhm)
        
        assert isinstance(smooth_f, pr.classes.Field)
        assert smooth_f.D == D
        assert smooth_f.fieldsize == dim + (nsubj,)
    
def test_wfield():
    nsubj = 10;
    for I in np.arange(2):
        if I == 0:
            dim = 5
            D = 1
        else:
            dim = (5,5)
            D = 2
            
        f = pr.wfield(dim, nsubj)
        assert isinstance(f, pr.classes.Field)

        assert f.D == D
        if I == 0:
            assert f.fieldsize == (dim, nsubj)
        else:
            assert f.fieldsize == dim + (nsubj,)

    
def test_statnoise():
    nsubj = 10;
    for I in np.arange(2):
        if I == 0:
            dim = 5
            D = 1
        else:
            dim = (5,5)
            D = 2
            
        fwhm = 4
        f = pr.statnoise(dim, nsubj, fwhm)
        assert isinstance(f, pr.classes.Field)

        assert f.D == D
        if I == 0:
            assert f.fieldsize == (dim, nsubj)
        else:
            assert f.fieldsize == dim + (nsubj,)
