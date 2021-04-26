import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import numpy as np
import pandas as pd
from scipy.special import softmax as scipy_softmax    

from set_device_dtype import *   
   





#####################################################################################################
#####################################################################################################







def log_sum_exp(x, dim):

    ''' Numerically stable log_sum_exp(x) 

    LSE = log_sum_exponent

    LSE(x1, x2, ... , xn) = log(  exp(x1) + exp(x2) + ... + exp(xn)  )
    LSE(x1, x2, ... , xn) = x* + log(  exp(x1-x*) + exp(x2-x*) + ... + exp(xn-x*)  )

    where x*  = max{x1, x2, ... , xn}

    '''

    x_star , idx = torch.max(x, dim=dim, keepdim=True)

    centered_x = x - x_star

    exp_centered_x = torch.exp(centered_x)

    log_sum_exp_centered_x = torch.log( torch.sum( exp_centered_x , dim=dim, keepdim=True) )

    lse = x_star + log_sum_exp_centered_x


    return lse




#####################################################################################################
#####################################################################################################





def log_softmax(x, dim):

    ''' Numerically stable log(softmax(x)) 

    x = [x1, x2, ... , xn ]

    softmax(x) = [ exp(x_1)/sum(exp(x)) , exp(x_2)/sum(exp(x)) , ... , exp(x_n)/sum(exp(x)) ]

    Note: Element-wise take natural log

    log( softmax(x) ) = log( [ exp(x_1)/sum(exp(x)) , exp(x_2)/sum(exp(x)) , ... , exp(x_n)/sum(exp(x)) ] )
                      =      [ x1 - LSE(x)   ,   x2 - LSE(x)  ,  ...  ,  xn - LSE(x)]

    '''

    x_star , idx = torch.max(x, dim=dim, keepdims=True)

    lse_x = log_sum_exp(x, dim=dim)

    log_softmax_x = x - lse_x

    return log_softmax_x    






#####################################################################################################
#####################################################################################################
    
    
    
    
def logistic_mixture_loss(model_output, target_data,  target_data_range, n_bins):

    ''' 


    Input:
    ------------------------------------------------------------------------------------------------
    1) model_output:        output of model after forward pass of input value                      :  shape (batch_size, n_target_nodes , n_mixture , 3)
    2) target_data:         target data to be used at the end of the network for loss calculation  :  shape (batch_size , 1 , n_target_nodes)
    3) target_data_range:   max(target_data) -  min(target_data)
    4) n_bins:              number of bins


    Output:
    ------------------------------------------------------------------------------------------------
    1) log-loss

    '''


    batch_size , n_target_nodes, n_mixtures , _  = model_output.shape   

    #print("Batch Size : " , batch_size)
    #print("N_mixtures : " , n_mixtures)


    # Extract out each of the mixture parameters 
    # model_output[ BatchSize , N_TargetNodes  , N_Mixtures  , 3]


    m             = model_output[ :, :, :, 0]    # mixture means
    log_scales    = model_output[ :, :, :, 1]    # mixture log_scales
    w_logits      = model_output[ :, :, :, 2]    # mixture raw weights, or logit_weights




    log_s = torch.clamp(log_scales , min=-7)    # imposing constraint on log_scales values, corresponding inv_s = e^(-log(s)) 
                                                #                                                               = e^(-(-7))
                                                #                                                               = 1096.63
                                                # Hence, minimum value of s that can be obtained is (1/1096.63) = 0.0009118


    inv_s = torch.exp(-log_s)                   # e^(-log(s)) = 1/s



    log_w = log_softmax(w_logits, dim=2)        # w_logits needs to be converted to valid weights that sum to one using softmax. also take natural_log od normalized weights



    x = target_data.reshape((batch_size , n_target_nodes , -1))            # shape ( batch_size , n_target_nodes , 1 )


    #print("model_output shape :" , model_output.shape)
    #print("Means shape        :" , m.shape)
    #print("Log_Scale shape    :" , log_scales.shape)
    #print("Logit_weight shape :" , w_logits.shape)
    #print("Target Data Shape  :" , x.shape)
    #print("Target Data        :" , x)    


    # There are total 'n_bins' number of bins. Bin index ranges from '0'  to  'n_bins-1'
    # 
    # range_of_train_data  =  max(train_data) - min(train_data)
    # When support of train_data is divided into equally spaced 'n_bins', then
    # width of one bin on the support of train_data is as follows ;

    bin_width      = target_data_range/(n_bins) 
    half_bin_width = bin_width/2.0



    # right and left points of bin around data point x 
    bin_arg_plus  =  ( x + half_bin_width - m )*inv_s   # shape ( batch_size , n_target_nodes, n_mixture )
    bin_arg_minus =  ( x - half_bin_width - m )*inv_s   # shape ( batch_size , n_target_nodes, n_mixture )




    # see documentation in markdown cell for derivation
    Px_i_mid_case  = torch.sigmoid(bin_arg_plus) - torch.sigmoid(bin_arg_minus) 



    # approximation of Px_i_mid_case when ( prob_mid_case << 1e-5 )   
    log_pdf = -(x-m)*inv_s - log_s - 2*F.softplus( -(x-m)*inv_s  )

    # approximate area under one-bin wide PDF curve = pdf*bin_width 
    # log(pdf * bin_width ) = log(pdf) + log( bin_width )    
    log_Px_i_mid_case_approximate = log_pdf + np.log(bin_width) 



    log_Px_i = torch.where( Px_i_mid_case > 1e-5,  torch.log(torch.clamp(Px_i_mid_case, min=1e-12)),                                             
                                                   log_Px_i_mid_case_approximate 
                          )


    alpha_i = log_w + log_Px_i 


    log_Px_k = log_sum_exp(alpha_i, dim=2) 


    loss = -torch.sum(log_Px_k)


    return loss  





#####################################################################################################
#####################################################################################################




def train( model ,     optimizer,                    loss_fn,                    n_mixtures,        n_bins,
                     train_data ,       input_col_indices=[],         target_col_indices=[],       root_node=False,
                  batch_size=500,             n_epochs=10000,              print_every= 500,       verbose=True):

    '''

    train_data target_data:        data to be used at the input and at the end of network (after forward) for loss calculation  :  shape ( batch_size , 1 , num_nodes )

    '''

    
    if verbose:
        # Total Number of Model Parameters
        total_params = sum(p.numel() for p in model.parameters())

        print("\n\n")
        print("Model/Neural Network :\n", model)
        print("\nNo. of Params :  ", total_params)
        print("\nOptimizer     :\n", optimizer)
        print("\nLoss Function : " , loss_fn)
        print("Num_mixtures    : " , n_mixtures)
        print("Num_Bins        : " , n_bins)
        print("Batch Size      : " , batch_size)
        print("Num_Epochs      : " , n_epochs)
        print("Input  Column Indices : " ,input_col_indices)
        print("Target Column Indices : " ,target_col_indices)
        print("\n\n\n")
        print('Training Started ...')
        print("\n")



    model = model.to(device=device)  # move the model parameters to CPU/GPU



    #                          Max of target nodes data             -        Min of target nodes data
    target_data_range = train_data[:,:,target_col_indices].max()    -   train_data[:,:,target_col_indices].min()


    num_samples = train_data.shape[0]      # Total number of examples in data set
    n_batches   = num_samples//batch_size  # Total number of batches in data set




    epoch_loss = []



    for epoch in range(n_epochs):


        batch_loss = []


        for batch_idx in range(n_batches):


            # Select batch of examples
            row_indices = np.arange( (batch_idx*batch_size), (batch_idx*batch_size+batch_size) , 1 )
            

            # For esitmation of root node's PDF, input to neural network will be zero. This (zero input) acts a dummy parent of root node of Bayesian network
            # Once model is trained, zero input will be used to get parameters of marginal PDF of root node.
            if root_node:
                model_input_batch= torch.zeros( ( len(row_indices), 1 , 1 ), dtype=dtype)
            else:
                model_input_batch= torch.from_numpy(train_data[row_indices][:,:,input_col_indices]).type(dtype)     # shape: [batch_size, 1 , ? ] 

            target_batch= torch.from_numpy(train_data[row_indices][:,:,target_col_indices])   # shape: [batch_size, 1 , ??] 



            model.train()                        # put model to training mode



            # Moving data to device CPU or GPU
            model_input_batch   =   (model_input_batch).to(device=device, dtype=dtype)  
            target_batch        =   (target_batch).to(device=device, dtype=dtype)



            model_output = model(model_input_batch)  # parameters of PDF(child_node | parents)   shape: ( batch_size , n_target_nodes , n_mol_comps , 3 )


            loss = loss_fn(model_output, target_batch, target_data_range, n_bins)


            batch_loss.append(loss.item())


            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()


            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()


            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass. 
            optimizer.step()


        epoch_loss.append(np.mean(batch_loss))


        if (epoch+1) % print_every == 0:
            print('Epoch No. %7d,     Loss = %.4f ' % (epoch+1, epoch_loss[-1] ))



    return epoch_loss




#####################################################################################################
#####################################################################################################



def get_valid_logistic_dist_params(model_output):

    """
    Extracts Logistic Mixture Density Parameters from the Output of NeuralNet

    Note: Neural network output contains means, natural_log_scales and un-normalized weights (logits) 



    Input
    ---------------------------------------------------------------------------
    1) model_output            shape: ( BatchSize , Num_Mixture_Components , 3 ) 


                               Data in last two dimensions:
                   |           ---------------------------------------------------------------------------          
                   v           |        1st Column     |       2nd Column      |       3rd Column        |
          Mixture Components   |-----------------------|-----------------------|-------------------------|
                               |          Means        |   Natural_Log(Scales) |  Un-normalized Weights  |
                               |           ...         |          ...          |             ...         |
                               ---------------------------------------------------------------------------




    Output
    ---------------------------------------------------------------------------
    1) valid PDF parameters    shape: (BatchSize, Num_Mixture_Components, 3)

                               Data in last two dimensions:
                   |           ---------------------------------------------------------------------------          
                   v           |        1st Column     |       2nd Column      |       3rd Column        |
          Mixture Components   |-----------------------|-----------------------|-------------------------|
                               |          Means        |        Scales         |     Normalized Weights  |
                               |           ...         |          ...          |             ...         |
                               ---------------------------------------------------------------------------

    """

    valid_pdf_params = np.zeros((model_output.shape))

    valid_pdf_params[:,:,0]   = model_output[:,:,0]                            # Extract means from 1st Column
    valid_pdf_params[:,:,1]   = np.exp(model_output[:,:,1])                    # Convert natural_log_scales to scales
    valid_pdf_params[:,:,2]   = scipy_softmax( model_output[:,:,2], axis=1 )   # Convert un-normalized weights (logits) to valid weights that sum to one

    return valid_pdf_params
