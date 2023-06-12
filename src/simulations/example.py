#!/usr/bin/env python3
# coding: utf-8

# imports 
import numpy as np
import matplotlib.pyplot as plt
from d_CSL import *

# select an integer `n_neur` and `l` to denote the number of variables and length of the variabales
n_neur = int(input())
l = int(innput())

# genderate a random ground truth conneectivity matrix A

A = adj_mtx(n_neur)

# simulate data 
noise = continuous_noise_fun(n_neur, l)
X = simulate_data()