import numpy as np


class BayesNetwork():
    
    def __init__(self, bn_structure):

        '''
        A class to create a Bayesian network

        Input:
        ------------------------------------------
        structure of Bayesian network should be given as dictionary. 
        keys of dictionary should be names of nodes (as string) and 
        values against each key or node should be a list of parents of that node (as a list of strings)

        assumption about structure of Bayesian network: no node appears as parent of another node until it has been registered in previous keys of dictionary 
        (forward sampling or ancestral sampling)

        '''
        

        
        
        self.nodes = list(bn_structure.keys())             # keys in 'bn_structure' dictionary contains names of nodes
        self.structure = bn_structure
        
        

        self.all_root_nodes   = self.all_root_nodes()      # all nodes that are root nodes in Bayesian network
        self.all_leaf_nodes   = self.all_leaf_nodes()      # all nodes that are leaf nodes in Bayesian network
        

        
        # check if the structure of Bayesian netwrok is suitable for forward/ancestral sampling.
        # if not, raise error with description
        self.isAncestralOrder(bn_structure)
    
        self.top_sorted_nodes = self.nodes

        self.all_parent_nodes = self.all_parent_nodes()    # all nodes that are parents in Bayesian network
        self.all_child_nodes  = self.all_child_nodes()     # all nodes that are children in Bayesian network
        

         
             
            
            
        # A dictionary containing the names of nodes in the form of integers
        # Nodes in self.nodes are mapped to [1,2,3,...N] where N is the number of nodes and corresponds to last node in self.nodes
        self.bn_numeric = {}    
        self.bn_numeric['nodes'] = np.arange(1, len(self.nodes)+1) 
        bn_structure_numeric = {}
        for node_numeric , node in enumerate(self.nodes,1):
            if node in self.all_root_nodes:
                bn_structure_numeric[node_numeric] = []  
            else:
                bn_structure_numeric[node_numeric] = list( map(self.node2index, self.parents(node) ) )
                
        self.bn_numeric['structure'] = bn_structure_numeric
        self.bn_numeric['all_parent_nodes'] =  list( map(self.node2index, self.all_parent_nodes ) ) 
        self.bn_numeric['all_child_nodes']  =  list( map(self.node2index, self.all_child_nodes  ) )
        self.bn_numeric['all_root_nodes']   =  list( map(self.node2index, self.all_root_nodes   ) )
        self.bn_numeric['all_leaf_nodes']   =  list( map(self.node2index, self.all_leaf_nodes   ) )
        
        
       
    
    
    def node2index(self, node):
        return self.nodes.index(node)+1   
        # assuming numeric labels of nodes start from 1 
        # e.g. for nodes ['A','B','C'], corresponding numeric labels will be [1,2,3]
    
    
        
    def isParent(self, node):
        '''
        To check if the given node is a parent node
        '''
        
        isParentFlag=False
        
        if len(self.structure[node])==0:
            isParentFlag=True
        else:
                       
            for nodee in self.nodes:
                isParentFlag = isParentFlag or (node in self.structure[nodee])
                if isParentFlag==True:
                    break
                
        return isParentFlag           
                
              
    
    def isChild(self,node):
        '''
        To check if the given node is a child node
        '''
        return len(self.structure[node])!=0
        
       
    
    def parents(self,node):
        '''
        To get parents of given node
        '''
        return self.structure[node]
    
    
    
    
    def children(self,node):
        '''
        To get children of given node
        '''
        children = set()
        
        # loop over all nodes and check if the given node (in argument of function) exists in the parents of each other node
        for nodee in self.nodes:
            if node in self.parents(nodee):
                children.add(nodee)
                
        return list(children)         
            
      
    
    
    def all_parent_nodes(self):
        '''
        To get a list of all parent nodes
        '''
        
        all_parents = []
        
        for node in self.nodes:
            if self.isParent(node):
                all_parents.append(node)
        
        
        return all_parents
    
    
    
    
    
    def all_child_nodes(self):
        '''
        To get a list of all child nodes
        '''
        
        all_children = []
        
        for node in self.nodes:
            if self.isChild(node):
                all_children.append(node)
        
        
        return all_children   
    

    
    
    
    def all_root_nodes(self):
        '''
        To get a list of all root nodes
        '''
        
        all_root_nodes = []
        
        for node in self.nodes:
            if len(self.parents(node))==0:
                all_root_nodes.append(node)
        
        
        return all_root_nodes   
       
        
        
        
        
    def all_leaf_nodes(self):
        '''
        To get a list of all leaf nodes
        '''
        
        all_leaf_nodes = []
        
        for node in self.nodes:
            if len(self.children(node))==0:
                all_leaf_nodes.append(node)
        
        
        return all_leaf_nodes 



    def isAncestralOrder(self, bn_structure):

        '''
        This function checks if the structure of Bayesian netwrok is suitable for forward/ancestral sampling.
        '''

        nodes = list(bn_structure.keys())  # keys in 'bn_structure' dictionary contains names of nodes

        for i, node in enumerate(nodes):

            if node in self.all_root_nodes: # if root node, skip
                pass
            else:                           # if child node, check if its parents have already been declared in previous nodes i.e. in nodes[:i]
                node_parents = bn_structure[node]
                for parent in node_parents:
                    if parent not in nodes[:i]:
                        raise Exception('''
Found: %s | %s                       	
Parent (%s) of node (%s) appeared before it was declared in the structure of Bayesian network before node (%s).
Structure of Bayesian network should be defined in a way that is consistent with forward/ancestral sampling.\n            
'''%(node, node_parents, parent,node,node) )
        
        
        # if everything is fine, True will be returned.
        return True
        	    	


    