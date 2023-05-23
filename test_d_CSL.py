#!/usr/bin/env python
# coding: utf-8


# import sys
# sys.path.append(r'./test_data/')
import d_CSL
import numpy as np

###################################################################
## tests to check the core functions the algorithm is dependent on
###################################################################


X = np.arange(30).reshape((3,10))
x,y,n_perm,n_past,n_lags = X[0],X[2],1000,2,2
prep_data = np.load(r'./test_data/test_prep_data.npy')
conditioning_set = np.load(r'./test_data/test_conditioning_set.npy')
np.random.seed(42)

def test_cross_corr():
    assert d_CSL.cross_corr(x, y, n_lags).all() == np.array([1., 1., 1.]).all()

def test_perm_test():
    x_,y_ = np.random.randn(1000), 2.5+ 2*np.random.randn(1000)
    assert round(d_CSL.perm_test(x_,y_,100000),1) == 0.2

def test_perm_test_shift():
    x_ = np.zeros(1000)
    x_[0] = np.random.randn()
    for i in range(1,len(x_)):
        x_[i] = x_[i-1] + np.random.randn()
    y_ = 2.5 + 2*x_
    assert d_CSL.perm_test_shift(x_,y_,100000) <= np.allclose(0.001)

def test_prep_data():
    assert d_CSL.prep_data(X,0).all() == X.all()
    assert d_CSL.prep_data(X,n_past).all() == prep_data.all()

def test_conditioning_set():
    assert d_CSL.conditioning_set(X,0,0,2).all() == X[1].all()
    assert d_CSL.conditioning_set(X,n_past,6,2).all() == conditioning_set.all()
    
def test_combine():
    assert np.array(combine([1,2,4,5,8,9,11])).all() == np.array([1,2,3,4,5,8,9,10,11]).all()
    assert np.array(combine([0,3,8,9,11])).all() == np.array([0,3,8,9,10,11]).all()