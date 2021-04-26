import numpy as np
import pandas as pd


    
def bayes_net_generate_samples(n_samples=100000, bayes_net_type=''):




    if bayes_net_type=='abcd':    # Structure of Bayesian Network:   a , b|(a) , c|(a,b) , d|(a,b,c)  

        # There are 4 nodes in Bayesian network.
        data = np.zeros( ( n_samples , 4), dtype=np.float32 )


        # Shape: ( num_samples, num_nodes, num_mix_comps , 3 )
        dist_params = np.zeros( (n_samples , 4, 3,3), dtype=np.float32 )


        # Names of columns of 2D-data matrix
        node_to_index = {}
        node_to_index['A']=0
        node_to_index['B']=1
        node_to_index['C']=2
        node_to_index['D']=3



        # Generating Samples of  A, B|(A) , C|(A,B) and D|(A,B,C) 
        for i in range(n_samples):



          
            ########################
            # Sampling from f(A)
         
            node_index=0
            dist_params[i,node_index,0,:]=[-10    ,0.5   ,0.3]     # [mean, scale, weight]
            dist_params[i,node_index,1,:]=[  0    ,0.2   ,0.3]
            dist_params[i,node_index,2,:]=[ 10    ,0.5   ,0.4]

            # Select mixture component (1st, 2nd or 3rd)
            mix_index = np.random.choice(3, 1, p=dist_params[i,node_index,:,2])   # mixture index (0,1, or 2)

            # Draw sample from selected mixture
            a = np.random.logistic(  dist_params[i,node_index,mix_index,0], dist_params[i, node_index ,mix_index,1] )
    





            ########################
            # Sampling from f(B|A=a)

            node_index=1
            dist_params[i,node_index,0,:]=[np.sqrt(np.abs(a))   ,np.sqrt(np.abs(a)+0.1)   ,0.4]  # [mean, scale, weight]
            dist_params[i,node_index,1,:]=[ a+10                ,np.sqrt(np.abs(a)+0.1)   ,0.2]
            dist_params[i,node_index,2,:]=[ 2*a-10              ,np.sqrt(np.abs(a)+0.1)   ,0.4]

            # Select mixture component (1st, 2nd or 3rd)
            mix_index = np.random.choice(3, 1, p=dist_params[i,node_index,:,2])   # mixture index (0,1, or 2)

            # Draw sample from selected mixture
            b = np.random.logistic(  dist_params[i,node_index,mix_index,0], dist_params[i, node_index ,mix_index,1] )
            
 



            ########################
            # Sampling from f(C|A=a,B=b)

            node_index=2
            dist_params[i,node_index,0,:]=[np.sqrt(np.abs(a))   ,np.sqrt(np.abs(a)+0.1)  ,0.3]  # [mean, scale, weight]
            dist_params[i,node_index,1,:]=[ 2*a+10              ,np.sqrt(np.abs(a)+0.1)  ,0.4]
            dist_params[i,node_index,2,:]=[ a+np.sin(b)-10      ,np.sqrt(np.abs(b)+0.1)  ,0.3]
            
            # Select mixture component (1st, 2nd or 3rd)
            mix_index = np.random.choice(3, 1, p=dist_params[i,node_index,:,2])   # mixture index (0,1, or 2)

            # Draw sample from selected mixture
            c = np.random.logistic(  dist_params[i,node_index,mix_index,0], dist_params[i, node_index ,mix_index,1] )
        




            ########################
            # Sampling from f(D|A=a,B=b,C=c)

            node_index=3
            dist_params[i,node_index,0,:]=[np.sqrt(np.abs(a+np.cos(c**2)))    ,np.sqrt(np.abs(a+b)+0.1),0.3]  # [mean, scale, weight]
            dist_params[i,node_index,1,:]=[ np.sin(2*b)-a +15                 ,np.sqrt(np.abs(a-c)+0.1),0.4]
            dist_params[i,node_index,2,:]=[ np.cos(a)+np.sin(c)-15            ,np.sqrt(np.abs(b-c)+0.1),0.3]


            # Select mixture component (1st, 2nd or 3rd)
            mix_index = np.random.choice(3, 1, p=dist_params[i,node_index,:,2])   # mixture index (0,1, or 2)

            # Draw sample from selected mixture
            d = np.random.logistic(  dist_params[i,node_index,mix_index,0], dist_params[i, node_index ,mix_index,1] )
 



            data[i,0]=a  
            data[i,1]=b 
            data[i,2]=c 
            data[i,3]=d 


 

        return data, dist_params, node_to_index



