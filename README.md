# Density Estimation in Continuous Bayesian Network with Neural Networks
This repository contains a neural-network based density estimator for a continuous Bayesian network.

This tool assumes all nodes in Bayesian network are continious and structure of Bayesian network is known.

Input: 
1) A dataframe in which each named column contains data of corresponding node.
2) A dictionary containing structure of Bayesian network


Output:
Each trained neural network will output parameters of conditional PDF of node X_i i.e. $`f(X_i | pa(X_i))`$



