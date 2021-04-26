
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import numpy as np
import pandas as pd
from scipy.special import softmax as scipy_softmax

# python files
from neural_net import get_neural_nets
from bayes_net import BayesNetwork
from logistic_mixture_loss import *
from helper_functions import *
from set_device_dtype import *   
from inference import *


class BayesNetDensityEstimator():
    
    """ 
    A class to learn marginal probability density function of root nodes and 
    conditional probability density function of child nodes in Bayesian network.
    
    """
    
    
    
    def __init__(self, nets_hidden_layer_sizes=[], bayes_net_structure={} ):
        
        
        
 
        self.bayes_net = BayesNetwork(bayes_net_structure)
        
        self.nodes = list(bayes_net_structure.keys())                      # keys in 'bayes_net_structure' dictionary contains names of nodes
         
        self.nets_hidden_layer_sizes = nets_hidden_layer_sizes             # sizes of hidden layers in neural network (each node will have its own neural network)      
     
        
        
        
    #####################################################################################################
    #####################################################################################################
        
      
    def arrange_df_nodes(self, data_frame_nodes, node):
        '''
        It re-arranges column indices of data frame such that they follow the order of parents of given node in self.bayes_net.structure[node]
        target_column_index is the index of column that contins data of given 'node'.
        
        '''
        input_column_indices = []
        node_parents = self.bayes_net.structure[node]

        for node_parent in node_parents:
            input_column_indices.append( data_frame_nodes.index(node_parent) ) 
           

        target_column_index = [data_frame_nodes.index(node)]
           
        return input_column_indices, target_column_index
        
        
        
        
        
    #####################################################################################################
    #####################################################################################################
        
    
    
    
    def learn_prob_density_params(self,  train_data=None, n_mol_comps=5, n_bins=500,  batch_size=500,  n_epochs=1000, print_every= 100, learning_rate=1e-2 , verbose=True):
        '''
        Input:
        ------------------------------------------------------------
        1) train_data:                            a data frame containing data of nodes of Bayesian network. Keys of data frame should be same as nodes in Bayesian network, regardless of order                      
        2) n_mol_comps:                           number of logistic distributions in mixture. Marginal PDF or conditional PDF  will contain "n_mol_comps" logistic distributions
        3) n_bins:                                number of intervals used to bin the PDF. The higher, more fine PDF bins 
        4) batch_size:                            batch size used during training
        
        
        
        '''
		
		
        train_data = train_data.copy()
		
        if not isinstance(train_data, pd.DataFrame):
            raise Exception('''
                            Train data should be a panda DataFrame with keys same as in the nodes of Bayesian network.\n
                            Type of given train data = %s \n
                            '''%(type(train_data)) )
        
        if batch_size > train_data.shape[0]:
            raise Exception('''
                            Batch size should be equal to or less than the number of samples in data.\n
                            Batch size = %d \n
                            Number of examples in data= %d\n
                            '''%(batch_size,train_data.shape[0]) ) 
            
        
        
        
        self.n_mol_comps = n_mol_comps       # number of logistic density functions in the mixture 
        self.n_bins = n_bins                 # number of bins used to discretize continuous PDF into intervals of uniform width
        
        self.norm_data = {}

        # for node in train_data.keys():
        #     self.norm_data[node] = ( train_data[node].mean(), train_data[node].std() )
        #     train_data[node] -= self.norm_data[node][0]
        #     train_data[node] /= self.norm_data[node][1]

        # a dictionary of neural networks (each node of Bayesian network will have its own neural network)
        self.neural_nets = get_neural_nets( self.nets_hidden_layer_sizes ,  self.n_mol_comps , self.bayes_net)
        
        # initializaing optimizer of each neural network
        optimizers={}
        for node in self.nodes:
            optimizers[node] = optim.RMSprop(self.neural_nets[node].parameters(), lr=learning_rate )
            #optimizers[node] = optim.Adam(self.neural_nets[node].parameters(), lr=learning_rate)
            #optimizers[node] = optim.AdamW(self.neural_nets[node].parameters(), lr=learning_rate)
        


        data_frame_nodes = list(train_data.keys()) 
        
        # check if the keys of data_frame and nodes in the structure of Bayesian network match or not
        import collections
        nodes_match_flag = collections.Counter(data_frame_nodes) == collections.Counter(self.nodes)

        if nodes_match_flag==False:
            raise Exception('''
                            Nodes in Bayesian network do not match with the keys of given data frame.\n
                            Nodes in Bayesian network and keys of given data frame should be same, regardless of order.\n
                            Nodes in Bayesian network = %s \n
                            Keys of data frame = %s
                            '''%(self.nodes, data_frame_nodes))
            
                                             
          


        # Training neural networks
        epoch_loss={}                    
        for node in self.nodes:


            root_node_flag= node in self.bayes_net.all_root_nodes
            if root_node_flag:
                print("\n\nLearning Parameters of Probability Density Function of  '%s' :\n"%(node))
            else:
                print("\n\nLearning Parameters of Probability Density Function of  '%s' | %s :\n"%(node, tuple(self.bayes_net.structure[node])))

            print('-----------------------------------------------------------------')
            print('-----------------------------------------------------------------\n')




            # ordered_input_column_indices =  indices of data frame arranged according to order of parents of given node i.e. self.bayes_net.structure[node]
            # target_column_index = target_column_index is the index of column of data_frame that contins data of given 'node'.                 
            ordered_input_column_indices, target_column_index  = self.arrange_df_nodes(data_frame_nodes, node)


            epoch_loss[node] = train(self.neural_nets[node], optimizers[node],        logistic_mixture_loss,        self.n_mol_comps ,  self.n_bins , train_data.values[: ,None,:] ,                    
                                                             input_col_indices=ordered_input_column_indices,   target_col_indices=target_column_index, root_node=root_node_flag,
                                                             batch_size=batch_size,          n_epochs=n_epochs,        print_every=print_every,   verbose=True )
   






    #####################################################################################################
    #####################################################################################################




    
    def get_valid_logistic_dist_params(self, model_output):

        """
        Extracts logistic mixture density parameters from the output of NeuralNet

        Note: Neural network output contains means, natural_log_scales and un-normalized weights (logits) 



        Input
        ---------------------------------------------------------------------------
        1) model_output            Output of neural network,    shape: ( BatchSize , Num_Mixture_Components , 3 ) 


                                   Data in last two dimensions:
                       |           ---------------------------------------------------------------------------          
                       v           |        1st Column     |       2nd Column      |       3rd Column        |
              Mixture Components   |-----------------------|-----------------------|-------------------------|
                                   |          Means        |   Natural_Log(Scales) |  Un-normalized Weights  |
                                   |           ...         |          ...          |             ...         |
                                   ---------------------------------------------------------------------------




        Output
        ---------------------------------------------------------------------------
        Shape: (BatchSize, Num_Mixture_Components, 3)

                                   Data in last two dimensions:
                       |           ---------------------------------------------------------------------------          
                       v           |        1st Column     |       2nd Column      |       3rd Column        |
              Mixture Components   |-----------------------|-----------------------|-------------------------|
                                   |          Means        |        Scales         |     Normalized Weights  |
                                   |           ...         |          ...          |             ...         |
                                   ---------------------------------------------------------------------------

        """

        valid_pdf_params = np.zeros((model_output.shape))

        valid_pdf_params[:,:,0]   = model_output[:,:,0]                            # Extract Means from 1st Column
        
        scales , weights = model_output[:,:,1],model_output[:,:,2]
        scales[scales > 10.0] = 10.0
        scales[scales < -10.0] = -10.0

        valid_pdf_params[:,:,1]   = np.exp(scales)                    # Convert natural_log_scales to scales
        valid_pdf_params[:,:,2]   = scipy_softmax( model_output[:,:,2], axis=1 )   # Convert un-normalized weights (logits) to valid weights that sum to one


        return valid_pdf_params





    #####################################################################################################
    #####################################################################################################


    
    
    def get_prob_density_params(self, test_data=None ):



        
        if not isinstance(test_data, pd.DataFrame):
            raise Exception('''
                            Test data should be a panda DataFrame with keys same as in the nodes of Bayesian network.\n
                            Type of given test data = %s \n
                            '''%(type(test_data)) )
          
        data_frame_nodes = list(test_data.keys()) 
        
        #for node in test_data.keys():
        #    self.norm_data[node] = ( train_data[node].mean(), train_data[node].std() )
        #    test_data[node] -= self.norm_data[node][0]
        #    test_data[node] /= self.norm_data[node][1]
        
        # check if the keys of data_frame and nodes in the structure of Bayesian network match or not
        import collections
        nodes_match_flag = collections.Counter(data_frame_nodes) == collections.Counter(self.nodes)

        if nodes_match_flag==False:
            raise Exception('''
                            Nodes in Bayesian network do not match with the keys of given data frame.\n
                            Nodes in Bayesian network and keys of given data frame should be same, regardless of order.\n
                            Nodes in Bayesian network = %s \n
                            Keys of data frame = %s
                            '''%(self.nodes, data_frame_nodes))
            
                                             
        test_batch_size = test_data.shape[0]  

         
        # returning a named dictionary containing parameters of PDFs
        prob_density_params = {}

        for node in self.nodes:


            # ordered_input_column_indices =  indices of data frame arranged according to order of parents of given node i.e. self.bayes_net.structure[node]
            # target_column_index = target_column_index is the index of column of data_frame that contins data of given 'node'.                 
            ordered_input_column_indices, target_column_index  = self.arrange_df_nodes(data_frame_nodes, node)
           
            self.neural_nets[node].train(mode=False)


           
            if node in self.bayes_net.all_root_nodes:
                model_input= torch.zeros( ( test_batch_size , 1 , 1 ), dtype=dtype)
                model_output = self.neural_nets[node]( model_input ).detach().numpy()                 # output of NN will be [ BatchSize  , 1  , n_mol_comps , 3 ] 
                prob_density_params[node] = self.get_valid_logistic_dist_params(model_output[:,0])[0] 
                
            
            else: 
                model_input= torch.from_numpy(test_data.values[:,None,:][:,:,ordered_input_column_indices] ).type(dtype)   
                model_output = self.neural_nets[node]( model_input ).detach().numpy()                 # output of NN will be [ BatchSize  , 1  , n_mol_comps , 3 ] 
                prob_density_params[node] = self.get_valid_logistic_dist_params(model_output[:,0])    
                
        
        
        return prob_density_params




    #####################################################################################################
    #####################################################################################################
    
    


    def get_node_pdf_params(self, node, node_parents_data=[] ):

        '''
        Input:
        ----------------------------------------
        1) node:                      node name
        2) node_parents_data:         an array of shape (BatchSize,1,N) containing data of given node's parents. 

        Caution: order of nodes in node_parents_data array should be same as in the 'self.bayes_net.structure[node]'

        Output:
        ----------------------------------------
        1) a matrix of shape (BatchSize, M, 3) containing parameters of PDF of node|parents  where M is the number of components in the mixture of logistic distributions
        '''

        self.neural_nets[node].train(mode=False)
        model_output = self.neural_nets[node]( torch.from_numpy(node_parents_data).type(dtype) ).detach().numpy() 
        prob_density_params = self.get_valid_logistic_dist_params(model_output[:,0])
        

        return prob_density_params      
  



    #####################################################################################################
    #####################################################################################################
    



    def draw_samples(self, n_samples=100 , do={} ):
        

        '''
        draw samples from trained neural networks. it also supports do() operation
        do:   a dictionary containing user-specified parameters of PDF node/s  
        '''
    

        samples_df = pd.DataFrame( np.zeros( (n_samples, len(self.nodes)), dtype=np.float32), columns=self.nodes )

        
        for node in self.nodes:

            # if user wants to perform do operation
            if node in do.keys(): 
                samples_df[node] = get_samples_from_logistic_mixture(  param_matrix=do[node], num_samps=n_samples  )

            # otherwise draw samples from parameters obtained from neural network
            else:
                if node in self.bayes_net.all_root_nodes:
                    node_parents_data=np.array([[[0]]])   # parameters of root-node's PDF are obtained by passing zero-input to neural network
                else:
                    node_parents = self.bayes_net.structure[node]
                    node_parents_data=samples_df[node_parents].values.reshape(n_samples,1,-1) 
               
                param_matrix=self.get_node_pdf_params( node, node_parents_data=node_parents_data )  # parameters of PDF(child_node|parents)


                samples_df[node] = get_samples_from_logistic_mixture( param_matrix=param_matrix, num_samps=n_samples )
               
    
        return samples_df
        





    #####################################################################################################
    #####################################################################################################



