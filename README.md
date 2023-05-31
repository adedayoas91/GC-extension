# GC-extension

This repo implements some enhancements we discussed in the paper currently in writing. We interpreted GC from Causal Bayesian networks (CBNs). 
GC as a framework does not take into account the problems of latent confounders. During our study, we found a way to identify spurious links induced by latent confounders by testing for Markovianity. We have successfully showed this worked on synthetic data, and have deployed it on onservational data of larva zebrafish. 

We found that the data is not Markov, and we are able to remove adequate number of inferred connections with our tests.

We implemented this algorithm as an open source and making it available for use in the neuroscience community.

To use this 

Dependencies:
 > Clone this repo to your local machine 
 > Run `make install` from terminal. If you prefer to use an anaconda distribution; Open the  `init.ipynb` notebook and run the cell. 
  This will install all required packages in the `requirements.txt` file.
 > An example notebook on how to use the package is also provided for you. 
> 