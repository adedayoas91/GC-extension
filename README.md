# GC-extension

This repo implements some enhancements we discussed in the paper currently in writing. We interpreted GC from Causal Bayesian networks (CBNs). 
GC as a framework does not take into account the problems of latent confounders. During our study, we found a way to identify spurios links induced by latent confounders by testing for Markovianity. We have successfully showed this worked on synthetic data, and have deployed it on onservational data of larva zebrafish. 

We found that the data is not Markov and we are able to remove adequate number of inferred connections with out tests.

We implemented this algorithm as an open source and making it available for use in the neuroscience community.

To use this 
