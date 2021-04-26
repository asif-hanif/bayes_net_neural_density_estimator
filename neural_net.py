import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import numpy as np
import pandas as pd

from scipy.special import softmax as scipy_softmax
  



class DenseNet(nn.Module):
    
    def __init__(self, net_layers_sizes, n_mol_comps):
        super().__init__()
        
        
        '''
        Input:
        ---------------------------------------------------
        1) net_layers_sizes        :    a list containing sizes of hidden layers in neural network
        2) n_mol_comps             :    number of components in the mixture of logsitc distributions for PDF of each node       
        '''


        self.n_mol_comps = n_mol_comps  
                      
        self.net = []
        for in_features, out_features in zip(net_layers_sizes, net_layers_sizes[1:]):

            self.net.extend([
                             nn.Linear(in_features, out_features),
                             #nn.BatchNorm1d(out_features),
                             nn.ReLU(),
                            ])

        self.net.pop() # pop the last ReLU for the output layer

        self.net = nn.Sequential(*self.net)


    def forward(self, x ):
     
        x = self.net(x)       
        x = x.view((-1, 1 , self.n_mol_comps , 3 ))
        
        '''
        If input's shape is  [BatchSize, 1, N_in]    (N_in is the number of parents of target node, 
        In case of root-node, N_in=1 because we assume a discrete dummy node as its parent) 
        Output of NN will be [ BatchSize  ,  1  , n_mol_comps , 3 ]
        For each logistic mixture component, there are three parameters: mean, scale, weight
        '''
        
        return x
   



def get_neural_nets(nets_hidden_layer_sizes, n_mol_comps,  bayes_network):

    '''
    Input:
    ------------------------------------------------
    1) nets_hidden_layer_sizes :    a dictionary containing sizes of neural network's hidden layers for each node
    2) n_mol_comps             :    number of components in the mixture of logsitc distributions for PDF of each node
    3) bayes_network           :    an object of class BayesianNetwork 
    '''
     
    
    
    nodes = bayes_network.nodes       # nodes in Bayesian network
    output_size = n_mol_comps*3       # number of neurons in the output layer of each neural network

    nets = {}
    
    
    for node in nodes:
        node_parents = bayes_network.structure[node]
        
        if node in bayes_network.all_root_nodes:
            net_layers_sizes = [1]+nets_hidden_layer_sizes[node] + [output_size]
        else:
            net_layers_sizes = [len(node_parents)] + nets_hidden_layer_sizes[node] + [output_size]
           
        nets[node] = DenseNet(net_layers_sizes, n_mol_comps)
      
 
    return nets    
    
    
    
        
    


