import numpy as np
import pandas as pd
from helper_functions import *



       
    #####################################################################################################
    #####################################################################################################
 


def get_indices_conditional_data_cont(df , conditions , verbose=False):

    """
    Purpose: To get conditional data.

    Evidence/Condition will be of the following form;
    [ (x_1<= EvidenceNode_i <=x_2)  AND  (x_3<= EvidenceNode_j <=x_4)  AND  ...  AND  (x_5<= EvidenceNode_k <=x_6)  ]


    Input
    ---------------------------------------------------------------------------
    1) df:              data_frame (continuous data)
    2) conditions:      conditions is a list of lists, each sublist contains [ evidence_node_name, lower_limit, upper_limit ]
 

    Output
    ---------------------------------------------------------------------------
    1) indices of the data_frame rows where evidence/conditions hold

    """

    bools = [True]*df.shape[0]


    evidence_not_found = {}
    # dictionary to save nodes whose ranges are not found in data

    if(len(conditions)==0):
        # If no evidence/condition is given, return unconditional data
        cond_data = df.loc[bools]

        if verbose:
            print('Evidence is empty. Returning un-conditional data.')

    else:

        # after this loop, bools will contain True at locations where given evidence exists in data frame
        for i in range(len(conditions)):     # conditions is a list of lists, each sublist contains [ node_name, lower_limit, upper_limit ]
            
            node            = conditions[i][0]
            low_limit       = conditions[i][1]
            up_limit        = conditions[i][2]

            bools_node = ( (df[node]>=low_limit) & (df[node]<=up_limit) )
            
            # if the range of a node's data, provided by user, is not found in actual data, save relevant data 
            if not any(bools_node):
                evidence_not_found[node] = [low_limit, up_limit, df[node].min(),  df[node].max()]


            bools = bools_node & bools

 


    if verbose:
        # description of conditions or evidence
        evid_desc = '[\n('

        for i in range(len(conditions)):       # conditions is a list of lists, each sublist contains;
                                               # [evidence_node_name, lower_limit,  upper_limit]
            node            = conditions[i][0]
            low_limit       = conditions[i][1]
            up_limit        = conditions[i][2]
            evid_desc +=  str(low_limit) + ' =< ' + str(node) + ' <= ' + str(up_limit) + ')\nAND\n( '

        evid_desc = evid_desc[:-7] + "\n]\n\n"
        print(evid_desc)


    return df[bools].index.values, evidence_not_found



       
    #####################################################################################################
    #####################################################################################################
 






def get_conditional_prob_cont(data_df=[] ,  query=[] , evidence=[] , verbose=False):


    """
    Purpose: To get conditional probability of continuous data.

    Evidence/Condition will be of the following form;
    [ (x_1<= EvidenceNode_i <=x_2)  AND  (x_3<= EvidenceNode_j <=x_4)  AND  ...  AND  (x_5<= EvidenceNode_k <=x_6)  ]

    and

    Query will be of the following form;
    [ (y_1<= QueryNode_i <=y_2)  AND  (y_3<= QueryNode_j <=y_4)  AND  ...  AND  (y_5<= QueryNode_k <=y_6)  ]



    Input
    ---------------------------------------------------------------------------
    data_df:            data_frame (continuous data)
    query:         query is a list of lists, each sublist contains [ qury_node_name, lower_limit, upper_limit ]
    evidence:      evidence is a list of lists, each sublist contains [ evidence_node_name, lower_limit, upper_limit ]



    Output
    ---------------------------------------------------------------------------
    probability of query node/s values lying in some range given an evidence

    """



    # get indices of those rows of data_frame where data of evidence nodes lie in their corresponding ranges 
    if verbose: print('Description of Evidence:')  
    evidence_data_indices , evidence_not_found = get_indices_conditional_data_cont(data_df ,  evidence , verbose=verbose)
    




    # if evidence is not found in data, raise error with description
    if bool(evidence_not_found):  # (if dictionary is not empty, it will evaluate to True)
        error_msg=""
        error_msg=error_msg+"\nProbablity: Following evidence was not found in data:\n"
        for node in list(evidence_not_found.keys()) :
            error_msg=error_msg+("Given: ( {:0.5f} =< "+node+" <= {:0.5f} ) ;    Actual: ({:0.5f} =< "+node+" <= {:0.5f})\n").format(evidence_not_found[node][0],evidence_not_found[node][1],evidence_not_found[node][2],evidence_not_found[node][3])
          
        raise Exception( error_msg )



    # get indices of those rows of data_frame where data of both query and evidence nodes lie in their corresponding ranges 
    if verbose: print('Description of Query:') 
    query_evidence_data_indices , _ = get_indices_conditional_data_cont(  data_df.iloc[evidence_data_indices] ,  query , verbose=verbose)



    prob = query_evidence_data_indices.shape[0]/evidence_data_indices.shape[0]


    return prob

   



    #####################################################################################################
    #####################################################################################################
 




def get_conditional_data_disc(data_df_disc , event_var, evidence_vars, evidence_vars_vals , verbose=False):
    
    
    """
    Purpose: To get conditional data of [event_node | evidence]
    
    
    Input
    ---------------------------------------------------------------------------
    data_df_disc:          data_frame containing discretized data of all nodes
    event_var:             character representing variable whose conditional probability is to be found
    evidence_vars:         list of characters representing variables used for evidence
    evidence_vars_vals:    list of integers representing values of evidence variables 


    Output
    ---------------------------------------------------------------------------
    Data[event_node | evidence]
    
    """
    
    
    bools = [True]*data_df_disc.shape[0]
    
    
    
    if(len(evidence_vars)==0):
        # If no evidence is given, return unconditional data of event variable
        event_var_cond_data = data_df_disc.loc[bools][event_var]
    
    
    else:
        # after this loop, bools will contain True at locations where given evidence exists in data frame
        for var, val in zip(evidence_vars, evidence_vars_vals):
            bools = bools & (data_df_disc[var]== val)


        # data of event variable conditioned on evidence
        # conditional data of event variable
        event_var_cond_data = data_df_disc.loc[bools][event_var]


        #print(event_var_cond_data)



        if verbose:
            # description of data
            data_desc = "ConditionalData[ %s | "%event_var

            for idx, v in enumerate(evidence_var):
                data_desc = data_desc + v + " = " + str(evidence_val[idx]) + " , "

            data_desc = data_desc[:-2] + "]"  
            print( data_desc  )

        
        
    return event_var_cond_data


  



    #####################################################################################################
    #####################################################################################################
 






def get_conditional_prob_disc(data_df_disc , event_var, event_var_vals, evidence_vars, evidence_vars_vals, verbose=False, nan2uniform=False):
   
    """
    Purpose: To get Prob[event_node | evidence]
    
    Input
    ---------------------------------------------------------------------------
    data_df_disc:          data_frame containing discretized data of all nodes
    event_var:             character representing variable whose conditional probability is to be found
    event_var_vals:        list of integers representing values/bins/states of event variable
    evidence_vars:         list of characters representing variables used for evidence
    evidence_vars_vals:    list of integers representing values of evidence variables
    nan2uniform:           if set to True and evidence is not found in data, function will return uniform distribution of event variable states 


    Output
    ---------------------------------------------------------------------------
    P[event_node | evidence]
    
    """
    
    
    # Obtain conditional data of event variable
    event_var_cond_data = get_conditional_data_disc(data_df_disc , event_var, evidence_vars, evidence_vars_vals, verbose=verbose)
    


    event_probs= np.zeros((len(event_var_vals)))


    # if evidence is not found and user wants the exception to be raised
    if len(event_var_cond_data) == 0 and nan2uniform==False:  
            raise Exception("Evidence not found in data.")

    # if evidence is not found and user wants the uniform distribution to be returned
    elif len(event_var_cond_data) == 0 and nan2uniform==True: 
         event_probs[:] = 1/len(event_var_vals)
    
    else:
        for i, val in enumerate(event_var_vals):
            event_probs[i] = len(event_var_cond_data[event_var_cond_data==val])/len(event_var_cond_data)

 

    return event_probs









    #####################################################################################################
    #####################################################################################################
 






def probs_from_logistic_interval_queries(interval, pdf_params):

    ''' 
    This function returns the sum of logistic interval queries.


    Input:
    ----------------------------------------
    1) interval:             a tuple (a,b) where a is lower end and b is upper end. 
                             For minus infinity use string '-inf' and for positive infinity use '+inf'

    2) pdf_params:          a matrix of shape (BatchSize, N,3) where N is the number of components. In each Nx3 matrix, 1st column= means, 2nd column=scales, 3rd column=weights
    '''



    probs=0
    M = pdf_params.shape[0]  # BatchSize

    for i in range(M):
        probs += logistic_interval_query( interval , param_matrix=pdf_params[i] ) 

    return probs




       
    #####################################################################################################
    #####################################################################################################
 




def cpquery_approx( query=None , evidence=None,  bnde=None, n_samples=1000, do={} , verbose=False  ):
    ''' 
    This function returns approximate probabiltiy of query|evidence using PDF parameters and MonteCarlo method. 


    Query will be of the following form;
    [ (y_1<= QueryNode_i <=y_2)  AND  (y_3<= QueryNode_j <=y_4)  AND  ...  AND  (y_5<= QueryNode_k <=y_6)  ]

    Evidence/Condition will be of the following form;
    [ (x_1<= EvidenceNode_i <=x_2)  AND  (x_3<= EvidenceNode_j <=x_4)  AND  ...  AND  (x_5<= EvidenceNode_k <=x_6)  ]



    Input
    ---------------------------------------------------------------------------
    query:              query is a list of lists, each sublist contains [ qury_node_name, lower_limit, upper_limit ]
    evidence:           evidence is a list of lists, each sublist contains [ evidence_node_name, lower_limit, upper_limit ]



    Output
    ---------------------------------------------------------------------------
    probability of query node/s values lying in their corresponding ranges given an evidence (if any) node/s values lying in their corresponding ranges



    '''

    

    query_nodes = [ query[i][0] for i in range(len(query)) ]               # nodes in query
    evid_nodes  = [ evidence[i][0] for i in range(len(evidence)) ]         # nodes in evidence (if any)

    query_evidence = query+evidence                                        # combined query and evidence
    query_evid_nodes = query_nodes+evid_nodes                              # combined nodes in query and evidence


    # deepest node in combined query_and_evidence nodes
    query_evid_deepest_node = get_deepest_node( query_evid_nodes , bnde.bayes_net.top_sorted_nodes  )  

    # combined query_and_evidence excluding a sublist containing deepest node
    query_evid_trimmed      = remove_sublist(query_evid_deepest_node, query_evidence)                       
      
    

    # draw samples from the trained neural nets (and do[?] if available)
    data_df = bnde.draw_samples( n_samples=n_samples, do=do )
    N = data_df.shape[0] # total number of samples






    ##########################
    #        Numerator      #
    ##########################


    # get indices of those rows of data_frame where data of query_evid_trimmed  nodes lie in their corresponding ranges 
    if verbose: print('Description of Query_Evidence_WithoutDeepestNode:') 
    query_evidence_trimmed_data_indices , evidence_not_found = get_indices_conditional_data_cont(  data_df ,  query_evid_trimmed , verbose=verbose)


    # if query_evid_trimmed is not found in data, raise error with description
    if bool(evidence_not_found):  # (if dictionary is not empty, it will evaluate to True)
        if (len(evidence) == 0):
            return 0.0

        error_msg=""
        error_msg=error_msg+"\nProbablity (Query_Evidence): Following conditions were not found in data:\n"
        for node in list(evidence_not_found.keys()) :
            error_msg=error_msg+("Given: ( {:0.5f} =< "+node+" <= {:0.5f} ) ;    Actual: ({:0.5f} =< "+node+" <= {:0.5f})\n").format(evidence_not_found[node][0],evidence_not_found[node][1],evidence_not_found[node][2],evidence_not_found[node][3])
          
        raise Exception( error_msg )
   


    query_evidence_trimmed_data= data_df.iloc[query_evidence_trimmed_data_indices]

    # compute numerator probability
    if query_evid_deepest_node in bnde.bayes_net.all_root_nodes:
        pdf_params = bnde.get_node_pdf_params( query_evid_deepest_node , node_parents_data=np.array([[[0]]]) )
        deepest_node_interval = tuple(find_sublist(query_evid_deepest_node, query_evidence)[1:3])
        prob_num = probs_from_logistic_interval_queries(deepest_node_interval, pdf_params) 

    else:
        node_parents = bnde.bayes_net.structure[query_evid_deepest_node]         # parents of deepest node
        node_parents_data = query_evidence_trimmed_data[node_parents].values     # values of parents of deepest node
        batch_size = node_parents_data.shape[0]
        # parameters of deepest_node given its parents
        pdf_params = bnde.get_node_pdf_params( query_evid_deepest_node , node_parents_data=node_parents_data.reshape(batch_size,1,-1) )  
        
        deepest_node_interval = tuple(find_sublist(query_evid_deepest_node, query_evidence)[1:3])
        prob_num = probs_from_logistic_interval_queries(deepest_node_interval, pdf_params) / N




    ##########################
    #       Denominator      #
    ##########################
    



    prob_den=1
    if len(evidence)!=0:

        evid_deepest_node = get_deepest_node( evid_nodes,        bnde.bayes_net.top_sorted_nodes  )    # deepest node in evidence nodes
        evidence_trimmed  = remove_sublist(evid_deepest_node, evidence)                                # evidence excluding a sublist containing deepest node


        
        # get indices of those rows of data_frame where data of evid_trimmed  nodes lie in their corresponding ranges 
        if verbose: print('Description of Evidence_WithoutDeepestNode:') 
        evidence_trimmed_data_indices , evidence_not_found = get_indices_conditional_data_cont(  data_df ,  evidence_trimmed , verbose=verbose)

        # if evid_trimmed is not found in data, raise error with description
        if bool(evidence_not_found):  # (if dictionary is not empty, it will evaluate to True)
            error_msg=""
            error_msg=error_msg+"\nProbablity (Evidence): Following  conditions were not found in data:\n"
            for node in list(evidence_not_found.keys()) :
                error_msg=error_msg+("Given: ( {:0.5f} =< "+node+" <= {:0.5f} ) ;    Actual: ({:0.5f} =< "+node+" <= {:0.5f})\n").format(evidence_not_found[node][0],evidence_not_found[node][1],evidence_not_found[node][2],evidence_not_found[node][3])
              
            raise Exception( error_msg )



        evidence_trimmed_data= data_df.iloc[evidence_trimmed_data_indices]

        # compute denominator probability 
        if evid_deepest_node in bnde.bayes_net.all_root_nodes:
            pdf_params = bnde.get_node_pdf_params( evid_deepest_node , node_parents_data=np.array([[[0]]]) )   
            deepest_node_interval = tuple(find_sublist(evid_deepest_node, evidence)[1:3])
            prob_den = probs_from_logistic_interval_queries(deepest_node_interval, pdf_params)
        else:    
            node_parents = bnde.bayes_net.structure[evid_deepest_node]      # parents of deepest node
            node_parents_data = evidence_trimmed_data[node_parents].values          # values of parents of deepest node
            batch_size = node_parents_data.shape[0]                         
            # parameters of deepest_node given its parents
            pdf_params = bnde.get_node_pdf_params( evid_deepest_node , node_parents_data=node_parents_data.reshape(batch_size,1,-1) )  
            
            deepest_node_interval = tuple(find_sublist(evid_deepest_node, evidence)[1:3])
            prob_den = probs_from_logistic_interval_queries(deepest_node_interval, pdf_params) / N
    

        # code to raise exception if prob_den is zero (to be added)


    return (prob_num/prob_den)




      
    #####################################################################################################
    #####################################################################################################
 




def get_deepest_node(test_nodes , top_sorted_nodes):

    '''
    Input:
    -------------------------------------------------
    1) test_nodes:           a list of nodes from which deepest node is to be found
    2) top_sorted_nodes:     topologically sorted nodes (list)

    Output:
    -------------------------------------------------
    1) a node (selected from test_nodes) that is at deepest level in topologically sorted Bayesian network
    '''

    test_nodes_idx = list(map(top_sorted_nodes.index, test_nodes))
    deepest_node_idx = np.argmax(test_nodes_idx)
    deepest_node = test_nodes[deepest_node_idx]

    return deepest_node
  



    #####################################################################################################
    #####################################################################################################
 


def find_sublist(node , list_of_sublists):

    '''
    Input:
    -------------------------------------------------
    1) node:                a node to be found in a list of sublists. each sublist consists of [node_name, lower_limit, upper_limit]
    1) list_of_sublists:    a list of sublists. each sublist consists of [node_name, lower_limit, upper_limit]

    Output:
    -------------------------------------------------
    1) a sublist in list_of_sublists whose first element matches with given node
    '''

    for i in range(len(list_of_sublists)):
        if list_of_sublists[i][0]==node:
            sublist = list_of_sublists[i]
            break
    

    return sublist    







      
    #####################################################################################################
    #####################################################################################################
 




def remove_sublist(node, list_of_sublists):
    '''
    Input:
    ----------------------------------------------
    1) node:                 a node (along with the sublist in which this node is present) to be removed from from a list_of_sublists(2nd argument)
    2) list_of_sublists:     list_of_sublists is a list of lists, each sublist contains [ node_name, lower_limit, upper_limit ]

    Output:
    -------------------------------------------------
    1) list_of_sublists excluding a sublist containing given node
    '''

    trimmed_list = list_of_sublists[:]   # creating a new object so that list_of_sublists remains unaffected

    for i in range(len(trimmed_list)):
        if trimmed_list[i][0]==node:
            del trimmed_list[i]
            break

    return trimmed_list      



