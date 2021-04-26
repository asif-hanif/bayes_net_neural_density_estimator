import numpy as np
from scipy.special import softmax as scipy_softmax
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import norm



################################################################################################################
################################################################################################################






def get_mix_logistic_pdf(means=None, scales=None, weights=None, param_matrix=None, support= np.linspace(-40,40,500) ):
    
    """
    Returns weighted sum (convex combination) of logistic probability density functions. 
    Support of all mixture components is same.
    
    
    Input
    ---------------------------------------------------------------------------
    Option-1  (means, scales and weights in separate arrays)
    a) means:                means of logistic distributions    ,  shape :  (N,)
    b) scales:               scales of logistic distributions   ,  shape :  (N,)
    c) weights:              weights of logistic distributions  ,  shape :  (N,)

    where N is the number of components


    Option-2  (means, scales and weights in one 2D matrix)
    a) param_matrix          a 2D matrix of shape (N,3) where N is the number of components:  1st column= means, 2nd column=scales, 3rd column=weights




    *) support:              support of logistic distributions  ,  shape :  (size,) 
    
    
    
    
    Output
    ---------------------------------------------------------------------------
    1) Support of Mixture of Logistic Distributions
    2) Evaluations of Mixture of Logistic Distributions at given support
    
    """


    
    if param_matrix is not None:
        means   = param_matrix[:,0]
        scales  = param_matrix[:,1]
        weights = param_matrix[:,2]

    n_mixtures = len(means)
    
    
    x = support
    
    
    x = x[ :,None]                 # shape: [n_smaples , 1]
    
    means   = means[None, :]       # shape: [1 , n_mixtures]
    scales  = scales[None, :]      # shape: [1 , n_mixtures]
    weights = weights[None, :]     # shape: [1 , n_mixtures]
    
  
        
    # Logistic Distribution
    # pdf(x)       = e^(-(x-m)/s) / {s(1 + e^{-(x-m)/s})^2}
    # log(pdf(x))  = -(x-m)/s - log(s) - log{ ( 1 + e^{-(x-m)/s} )^2 }
    # log(pdf(x))  = -(x-m)/s - log(s) - 2*log{ ( 1 + e^{-(x-m)/s} ) }
    
    
    log_pdfs = -(x-means)/scales - np.log(scales) - 2*np.log( 1 + np.exp(-(x-means)/scales)  )    
    
    weighted_log_pdfs = np.log(weights) + log_pdfs
    
    pdfs = np.exp(weighted_log_pdfs)
     
    pdf = np.sum( pdfs , axis=1 )    
        
    return support , pdf                        
  


    
    
    
    


################################################################################################################
################################################################################################################







def sigmoid(x):
    
    '''
    Sigmoid function
    
    Applies sigmoid function on single element x.
    
    '''
    
    # Protect from underflow
    if x < -20:
        return 0.0
    
    # Protect from overflow
    elif x > 20:
        return 1.0
    
    else:
        return np.exp(x)/(1+np.exp(x))

      

    
    
def logistic_cdf(x, mean, scale):
    
    '''
    CDF of logistic distribution
    P[ X <= x]
    This function assumes signle element at input.
    '''
    
    return sigmoid((x - mean) / (scale+1e-5))







################################################################################################################
################################################################################################################






def convert_logistic_pdf_to_probs( mean, scale , intervals= np.linspace(-40,40,500) ):

    
    '''
    Converts probability density function (logistic distribution) into probabilities by binning.
    PDF is binned according to given intervals and areas are calculated under these bins/intervals.
    
    For probability calculation, 
     1) first interval is assumed to be from -infinity to the left edge of 1st interval provided by the user.
     2) last interval is assumed to be from the right edge of last interval provided by the user to +infinity. 
    
    user_intervals  = [a , b , c , d]    : there are three intervals/bins in this array
    code_ intervals = [-infinity ,   a , b , c , d ,  +infinity ] : there are five intervals/bins in this array

    Following probabilities are returned;

    P[-infinity < X <= a] , P[a <= X <= b] , P[b <= X <= c] , P[c <= X <= d] , P[d <= X <= +infinity]



    Input
    ---------------------------------------------------------------------------
    1) mean:                mean of logistic distribution
    2) scale:               scale of logistic distribution
    3) intervals:           intervals or bins obtainied from slices of random variable's support 
    
    
    Note: If intervals is not given in argument, unifrom intervals are used from default value of argument.
    
    
    Output
    ---------------------------------------------------------------------------
    1) Bins/Intervals 
    2) Probabilities of Bins/Intervals
    '''
    
    probs = []
    
    
    

    # Minimum and Maximum Value of Support and Number of bins/intervals
    min_x    = intervals[0]
    max_x    = intervals[-1]
    num_bins = len(intervals)-1


    #bin_width      = (max_x - min_x)/(num_bins)
    #half_bin_width = bin_width/2.0
    #half_bin_width = 0.5


    x = intervals

    for i in range(len(x)):

        if i ==0 :           # left edge of first interval provided by user
            prob = logistic_cdf( x[i]  ,  mean , scale )  # P[-infinity < X <= a]  where a is left edge of first interval provided by user
            
            probs.append(prob)

        elif i== (len(x)-1): # right edge of last interval provided by user
            # P[c <= X <= d] where c is the left edge and d is the right edge of last interval provided by user
            prob = logistic_cdf( x[i]  ,  mean , scale ) - logistic_cdf( x[i-1]  ,  mean , scale )
            
            # P[d <= X <= +infinity]  where d is the right edge of last interval provided by user
            prob_infinty = 1 - logistic_cdf( x[i]  ,  mean , scale )  
            

            probs.append(prob)
            probs.append(prob_infinty)

        else:                # mid case
            prob = logistic_cdf( x[i]  ,  mean , scale ) - logistic_cdf( x[i-1]  ,  mean , scale )
            
            probs.append(prob)


        


    ## appending -infinity and +infinity to input intervals array
    #intervalss = np.zeros( (len(intervals)+2) )
    #intervalss[0] = -np.inf
    #intervalss[1:(len(intervals)+1)] = intervals
    #intervalss[-1] = +np.inf
 
    return intervals, probs





################################################################################################################
################################################################################################################





def convert_logistic_mixture_to_probs(means=None, scales=None, weights=None, param_matrix=None, intervals= np.linspace(-40,40,500)):
    
    '''
    Converts mixture of probability density functions (logistic distributions) into probabilities by binning.
    PDFs are binned according to given intervals and areas are calculated under these bins/intervals.
    The probabilites (returned by function) are obtained by taking convex combination of probabilities from each of the PDFs in mixture. 
    
    Input
    ---------------------------------------------------------------------------
    Option-1  (means, scales and weights in separate arrays)
    a) means:                means of logistic distributions    ,  shape :  (N,)
    b) scales:               scales of logistic distributions   ,  shape :  (N,)
    c) weights:              weights of logistic distributions  ,  shape :  (N,)

    where N is the number of components


    Option-2  (means, scales and weights in one 2D matrix)
    a) param_matrix          a 2D matrix of shape (N,3) where N is the number of components:  1st column= means, 2nd column=scales, 3rd column=weights


    *) intervals:            intervals or bins obtainied from slices of random variable's support,   shape: (size,) 
    
    
    Note: If 'intervals' is not given in argument, unifrom intervals are used from default value of argument.
    
    
    Output
    ---------------------------------------------------------------------------
    1) Bins/Intervals 
    2) Probabilities of Bins/Intervals
    '''
        
        
    if param_matrix is not None:
        means   = param_matrix[:,0]
        scales  = param_matrix[:,1]
        weights = param_matrix[:,2]


    n_mixtures  = len(means)    # Number of Mixture Components
    
    p_mixtures = []             # List for Probabilites Obtained from Each Mixture Component PDF
    
    
    # Loop over mixture components
    for i in range(n_mixtures):
        
        # Convert Probability Density Function (Logistic Distribution) into Probabilities by Binning
        _ , p_component_i = convert_logistic_pdf_to_probs( means[i], scales[i] , intervals= intervals )

        # Weight probabilites with corresponding mixture_weight
        p_mixtures.append(weights[i] * np.array(p_component_i) )
        
    
        
    return intervals, np.sum(p_mixtures, axis=0)   # Return weighted sum of mixture probabilities







################################################################################################################
################################################################################################################






def get_samples_from_logistic_mixture(means=None, scales=None, weights=None, param_matrix=None, num_samps=1):
    
    '''
    This function provides samples drawn from mixture of logisitic probability density functions.
    
    Input:
    -------------------------------------------------------------------------------
    Option-1  (means, scales and weights in separate arrays)
    a) means:                means of logistic distributions    ,  shape :  (N,)
    b) scales:               scales of logistic distributions   ,  shape :  (N,)
    c) weights:              weights of logistic distributions  ,  shape :  (N,)

    where N is the number of components


    Option-2  (means, scales and weights in one 2D matrix)
    a) param_matrix          a 3D matrix of shape (BatchSize, N,3) where N is the number of components. in each Nx3 matrix:  1st column= means, 2nd column=scales, 3rd column=weights


    *) num_samps:            number of samples required
  
  
    Output:
    -------------------------------------------------------------------------------
    1) smaples drawn from mixture of logistic probability density functions   ,  shape :  (num_samps,)
    
    '''

    samples = np.zeros((num_samps))

    if param_matrix is not None:
        means   = param_matrix[:,:,0]
        scales  = param_matrix[:,:,1]
        weights = param_matrix[:,:,2]

    else:
        means = np.array(means)[np.newaxis,:]
        scales = np.array(scales)[np.newaxis,:]
        weights = np.array(weights)[np.newaxis,:]




    # if sum(weights) is not equal to zero with pre-defined tolerance, np.random.choice() will raise value error.
    # normalize weights to make them sum to one
    if (np.sum(weights, axis=-1).any() < 1+0.005) & (np.sum(weights, axis=-1).any() > 1-0.005):
        weights = np.array(weights)/np.sum(weights, axis=-1, keepdims=True)

    num_mix = len(means[0,:])  # number of mixture components
    

    for i in range(num_samps):
           
        j = i if weights.shape[0]>1 else 0    # if root node, j=0 otherwise j=1
        
        # select mixture component using weights as probability
        w_index = np.random.choice(num_mix , 1, p=weights[j] ) 
        
        # sample from selected mixture logistic component
        samples[i] = np.random.logistic( means[j,w_index]  , scales[j,w_index]  ) 
     
    return samples





################################################################################################################
################################################################################################################








def logistic_interval_query( interval, means=None, scales=None, weights=None, param_matrix=None):
    
    '''
    Provides probability of an interval under mixture of probability density functions (logistic distributions). 
    Area under each PDF curve (and above the given interval) is calculated.
    Probabilites obtained from each of the PDFs are weighted with corresponding mixture weights. 
    
    Input
    ---------------------------------------------------------------------------
    1) interval:             a tuple (a,b) where a is lower end and b is upper end. 
                             For minus infinity use string '-inf' and for positive infinity use '+inf'
    


    Option-1  (means, scales and weights in separate arrays)
    a) means:                means of logistic distributions    ,  shape :  (N,)
    b) scales:               scales of logistic distributions   ,  shape :  (N,)
    c) weights:              weights of logistic distributions  ,  shape :  (N,)

    where N is the number of components


    Option-2  (means, scales and weights in one 2D matrix)
    a) param_matrix          a 2D matrix of shape (N,3) where N is the number of components:  1st column= means, 2nd column=scales, 3rd column=weights
    
    
   
    
    Output
    ---------------------------------------------------------------------------
    1) Probability of Bin/Interval,  P[ a <=  X  <= b ]
    '''
       
    if param_matrix is not None:
        means   = param_matrix[:,0]
        scales  = param_matrix[:,1]
        weights = param_matrix[:,2]

       
        
    a , b = interval    
    
    n_mixtures  = len(means)        # Number of Mixture Components
    
    probs_mixtures = []             # List for Probabilites Obtained from CDF of Each Mixture Component
    
    
    # Loop over mixture components
    for i in range(n_mixtures):
           
        
        if a=='-inf':
            P_ai=0                                        # P[ X <= -infinity]  under ith mixture component
        else: 
            P_ai = logistic_cdf(a, means[i], scales[i])   # P[ X <= a]          under ith mixture component
        
        
        
        if b=='+inf':
            P_bi = 1                                      # P[ X <= +infinity]  under ith mixture component
        else:
            P_bi = logistic_cdf(b, means[i], scales[i])   # P[ X <= b]          under ith mixture component
        
        
        
        P_abi = P_bi - P_ai                               # P[ a <=  X  <= b ]  under ith component
        
        
        # Weight probabilites with corresponding mixture_weight
        probs_mixtures.append(    weights[i] * np.array(P_abi) )
        
    
        
    return np.sum(probs_mixtures, axis=0)   # Return weighted sum of mixture probabilities






################################################################################################################
################################################################################################################




def TVD(P,Q):
    
    """
    Total Variation Distance between two distributions P and Q
    
    TVD  = 0.5 * SUM{ |P(x) - Q(x)| } for all x
    """
    
    diff = P-Q
    
    tvd = 0.5*np.sum(np.abs(diff))
    
    return tvd






################################################################################################################
################################################################################################################




def learn_LGM( structure, data_df):
    
    """
    Learns parameters of Linear-Gaussian-Model (LGM)

    1) structure:    a dictionary containing structure of Bayesian network
    2) data_df:      a panada DataFrame containing data of all nodes. All columns must be named with nodes.

    """
    
    


    lgm= {}
    

    for node in list(structure.keys()):

        pa_node = structure[node] # parents of node

        if len(pa_node)==0: # if root node

            lgm[node]= [ np.mean(data_df[node].values), np.std(data_df[node].values) ]  # mean and standard deviation of node values

        else:                # if child node

            reg_coeffs_node = LinearRegression().fit(  data_df[pa_node].values,  data_df[node].values   )   # regression coefficients 
            std_node = np.std( reg_coeffs_node.predict(data_df[pa_node].values) - data_df[node].values  )   # stanadard deviation of residuals

            lgm[node]= [reg_coeffs_node, std_node]
            

    return lgm






################################################################################################################
################################################################################################################






def convert_norm_pdf_to_probs(mean, scale , intervals= np.linspace(-40,40,500)):
    
    '''
    Converts probability density function (normal distribution) into probabilities by binning.
    PDF is binned according to given intervals and areas are calculated under these bins/intervals.
    
    For probability calculation, 
     1) first interval is assumed to be from -infinity to the left edge of 1st interval provided by the user.
     2) last interval is assumed to be from the right edge of last interval provided by the user to +infinity. 
    
    user_intervals  = [a , b , c , d]    : there are three intervals/bins in this array
    code_ intervals = [-infinity ,   a , b , c , d ,  +infinity ] : there are five intervals/bins in this array

    Following probabilities are returned;

    P[-infinity < X <= a] , P[a <= X <= b] , P[b <= X <= c] , P[c <= X <= d] , P[d <= X <= +infinity]



    Input
    ---------------------------------------------------------------------------
    1) mean:                mean of normal distribution
    2) scale:               standard deviation of normal distribution
    3) intervals:           intervals or bins obtainied from slices of random variable's support 
    
    
    Note: If intervals is not given in argument, unifrom intervals are used from default value of argument.
    
    
    Output
    ---------------------------------------------------------------------------
    1) Bins/Intervals 
    2) Probabilities of Bins/Intervals
    '''

    probs = []




    # Minimum and Maximum Value of Support and Number of bins/intervals
    min_x    = intervals[0]
    max_x    = intervals[-1]
    num_bins = len(intervals)-1


    #bin_width      = (max_x - min_x)/(num_bins)
    #half_bin_width = bin_width/2.0
    #half_bin_width = 0.5


    x = intervals
    

    for i in range(len(x)):

        if i ==0 :           # left edge of first interval provided by user
            prob = norm.cdf( x[i]  ,  loc=mean , scale=scale )  # P[-infinity < X <= a]  where a is left edge of first interval provided by user
            
            probs.append(prob)


        elif i== (len(x)-1): # right edge of last interval provided by user
            # P[c <= X <= d] where c is the left edge and d is the right edge of last interval provided by user
            prob = norm.cdf( x[i]  ,  loc=mean , scale=scale ) - norm.cdf( x[i-1] ,  loc=mean , scale=scale )
            
            # P[d <= X <= +infinity]  where d is the right edge of last interval provided by user
            prob_infinty = 1 - norm.cdf( x[i]  ,  loc=mean , scale=scale )  
            

            probs.append(prob)
            probs.append(prob_infinty)

        else:                # mid case
            prob = norm.cdf( x[i]  ,  loc=mean , scale=scale ) - norm.cdf( x[i-1] ,  loc=mean , scale=scale )
            
            probs.append(prob)

    


    ## appending -infinity and +infinity to input intervals array
    #intervalss = np.zeros( (len(intervals)+2) )
    #intervalss[0] = -np.inf
    #intervalss[1:(len(intervals)+1)] = intervals
    #intervalss[-1] = +np.inf
 

    return intervals , np.array(probs).squeeze()
