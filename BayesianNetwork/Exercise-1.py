from collections import defaultdict

import numpy as np


class Variable:

    def __init__(self, name, no_states, table, parents=[], no_parent_states=[]):
        """
        name (string): Name of the variable
        no_states (int): Number of states this variable can take
        table (list or Array of reals): Conditional probability table (see below)
        parents (list of strings): Name for each parent variable.
        no_parent_states (list of ints): Number of states that each parent variable can take.

        The table is a 2d array of size #events * #number_of_conditions.
        #number_of_conditions is the number of possible conditions (prod(no_parent_states))
        If the distribution is unconditional #number_of_conditions is 1.
        Each column represents a conditional distribution and sum to 1.

        Here is an example of a variable with 3 states and two parents cond0 and cond1,
        with 3 and 2 possible states respectively.
        +----------+----------+----------+----------+----------+----------+----------+
        |  cond0   | cond0(0) | cond0(1) | cond0(2) | cond0(0) | cond0(1) | cond0(2) |
        +----------+----------+----------+----------+----------+----------+----------+
        |  cond1   | cond1(0) | cond1(0) | cond1(0) | cond1(1) | cond1(1) | cond1(1) |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(0) |  0.2000  |  0.2000  |  0.7000  |  0.0000  |  0.2000  |  0.4000  |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(1) |  0.3000  |  0.8000  |  0.2000  |  0.0000  |  0.2000  |  0.4000  |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(2) |  0.5000  |  0.0000  |  0.1000  |  1.0000  |  0.6000  |  0.2000  |
        +----------+----------+----------+----------+----------+----------+----------+

        To create this table you would use the following parameters:

        Variable('event', 3, [[0.2, 0.2, 0.7, 0.0, 0.2, 0.4],
                              [0.3, 0.8, 0.2, 0.0, 0.2, 0.4],
                              [0.5, 0.0, 0.1, 1.0, 0.6, 0.2]],
                 parents=['cond0', 'cond1'],
                 no_parent_states=[3, 2])
        """
        self.name = name
        self.no_states = no_states
        self.table = np.array(table)
        self.parents = parents
        self.no_parent_states = no_parent_states

        if self.table.shape[0] != self.no_states:       #np.array.shape returns number of elements in each dimension, shape[0] = events here 
            raise ValueError(f"Number of states and number of rows in table must be equal. "
                             f"Recieved {self.no_states} number of states, but table has "
                             f"{self.table.shape[0]} number of rows.")

        if self.table.shape[1] != np.prod(no_parent_states):    #num of columns = combos of parents states, 6 here
            raise ValueError("Number of table columns does not match number of parent states combinations.")

        if not np.allclose(self.table.sum(axis=0), 1):          #Check if sum of all events given a condition equals 1
            raise ValueError("All columns in table must sum to 1.")

        if len(parents) != len(no_parent_states):               #One state for each parent
            raise ValueError("Number of parents must match number of length of list no_parent_states.")

    def __str__(self):
        """
        Pretty string for the table distribution
        For printing to display properly, don't use variable names with more than 7 characters
        """
        width = int(np.prod(self.no_parent_states))             #multiplies the elements in list, here tot num of states
        grid = np.meshgrid(*[range(i) for i in self.no_parent_states]) 
        s = ""
        for (i, e) in enumerate(self.parents):
            s += '+----------+' + '----------+' * width + '\n'
            gi = grid[i].reshape(-1)
            s += f'|{e:^10}|' + '|'.join([f'{e + "("+str(j)+")":^10}' for j in gi])
            s += '|\n'

        for i in range(self.no_states):
            s += '+----------+' + '----------+' * width + '\n'
            state_name = self.name + f'({i})'
            s += f'|{state_name:^10}|' + '|'.join([f'{p:^10.4f}' for p in self.table[i]])
            s += '|\n'

        s += '+----------+' + '----------+' * width + '\n'

        return s

    def probability(self, state, parentstates):
        """
        Returns probability of variable taking on a "state" given "parentstates"
        This method is a simple lookup in the conditional probability table, it does not calculate anything.

        Input:
            state: integer between 0 and no_states
            parentstates: dictionary of {'parent': state}
        Output:
            float with value between 0 and 1
        """
        if not isinstance(state, int):
            raise TypeError(f"Expected state to be of type int; got type {type(state)}.")
        if not isinstance(parentstates, dict):
            raise TypeError(f"Expected parentstates to be of type dict; got type {type(parentstates)}.")
        if state >= self.no_states:
            raise ValueError(f"Recieved state={state}; this variable's last state is {self.no_states - 1}.")
        if state < 0:
            raise ValueError(f"Recieved state={state}; state cannot be negative.")

        table_index = 0
        for variable in self.parents:
            if variable not in parentstates:
                raise ValueError(f"Variable {variable} does not have a defined value in parentstates.")
                        #TODO sjekk opp feil her, tydeligvis være variable istedenfor .name
            var_index = self.parents.index(variable)
            table_index += parentstates[variable] * np.prod(self.no_parent_states[:var_index])

        return self.table[state, int(table_index)]


class BayesianNetwork:
    """
    Class representing a Bayesian network.
    Nodes can be accessed through self.variables['variable_name'].
    Each node is a Variable.

    Edges are stored in a dictionary. A node's children can be accessed by
    self.edges[variable]. Both the key and value in this dictionary is a Variable.
    """
    def __init__(self):
        self.edges = defaultdict(lambda: [])  # All nodes start out with 0 edges
        self.variables = {}                   # Dictionary of "name":TabularDistribution

    def add_variable(self, variable):
        """
        Adds a variable to the network.
        """
        if not isinstance(variable, Variable):
            raise TypeError(f"Expected {Variable}; got {type(variable)}.")
        self.variables[variable.name] = variable

    def add_edge(self, from_variable, to_variable):
        """
        Adds an edge from one variable to another in the network. Both variables must have
        been added to the network before calling this method.
        """
        if from_variable not in self.variables.values():
            raise ValueError("Parent variable is not added to list of variables.")
        if to_variable not in self.variables.values():
            raise ValueError("Child variable is not added to list of variables.")
        self.edges[from_variable].append(to_variable)

    def sorted_nodes(self):
        """
        Returns: List of sorted variable names.
        """
        #The list returned at the end, topologically sorted
        resulting_nodes = []

        #all nodes in the structure
        all_nodes = list(self.variables.values())   # [objVal, objVal, ..], objVal = objectValue

        #nodes is as many times in this as number of incoming edges
        nodes_with_incoming_edge = list(self.edges.values())  #[[dest, dest, dest], [dest, dest], ..]
        
        #finds nodes with no incoming edges
        no_edge = all_nodes.copy()         # [node, node, ..]
        for node_incoming in nodes_with_incoming_edge:#starts with all nodes, removes the one with incoming edges
            for node in node_incoming:
                if node in all_nodes:
                    if node in no_edge:
                        no_edge.remove(node)        #the resulting nodes is the ones with no incoming edges, rest is removed

        #The structure, is manipulated during run by removing already visited nodes(moved to resulting_nodes)
        manipulate_edges = self.edges.copy() #dict{node:[dest, dest], node:[dest], ..}
        
        while no_edge != []:    #While there exist a node with no edges in
            current = no_edge.pop(0)            #node first in line with no incoming edges
            resulting_nodes.append(current)     #add to result
            if current in manipulate_edges:     #if existing in the structure, remove it. Can be that is not in structure if several pointers out
                manipulate_edges.pop(current) #gets the destinations from current
            new_struct = list(manipulate_edges.values())    #List of all nodes with incoming edges in new structure
            #print("\nnew valuestructure\n", new_struct, "\n\n") #TODO denna her endrer resultatet. fremdeles rett men endrer plass på b og c
            
            new_graph = manipulate_edges.copy()
            #print("new graph\n", new_graph)
            temp_all = all_nodes.copy()                #same method as before loop, assume all nodes have no incoming edges and removes the ones that does
            temp_all = set(temp_all) - set(resulting_nodes) #removes the nodes already visited and added to result
            for node_incoming in new_struct:
                for node in node_incoming:  #legg til alle noder som ikke er i new_struct!
                    if node in temp_all:
                        temp_all.remove(node)   #removes all the nodes that are destinations for edges
            no_edge.extend(temp_all)            #adds the new nodes with no incoming edges to no_edge list
            #print("no edges now:",no_edge,"\n\n")
        

        resulting_nodes.pop(-1)                     #The last element is added twice, remove last.
        print("\n\nThe resulting nodes:\n", resulting_nodes)
        print("\n", self.variables, "\n\n")
        
        return resulting_nodes              #Know this maybe got quite a bit more complicated than it needed to be... But for sure no cribbing:)) For sure it would be easier to use the information of each node, like self.parents but figured out this a bit to late...



class InferenceByEnumeration:
    def __init__(self, bayesian_network):
        self.bayesian_network = bayesian_network
        self.topo_order = bayesian_network.sorted_nodes()
    
    def _enumeration_ask(self, X, evidence):
        # Pretty much following the pseudocode
        # X = query variable , eg. 'C'
        # evidence = {'D':1}
        
        query_node = self.bayesian_network.variables[X] #the object querynode
        Q = np.zeros(query_node.no_states)  #Distribution over X, init empty with 0s

        for xi in range(query_node.no_states):  #one iteration for each state of the query node
            evidence_copy = evidence.copy()     #copy the evidence, ensure no manipulating in the recursive funcs
            evidence_copy[X] = xi               #adds the querynode with value to evidence
            Q[xi] = self._enumerate_all(self.topo_order.copy(), evidence_copy) # enumerates, recursive call to compute the probability of given state
        
        
        
        normalization_factor = 1/np.sum(Q)  #creates the normalizationfactor
        for i in range(len(Q)):
            Q[i] *= normalization_factor    #multiply normFactor into values
        #print(Q)
        return Q          #The distribution of Q, hopefully correct and sums up to 1 after normalization

    def _enumerate_all(self, vars1, evidence):

        depth_of_vars1 = len(vars1)         #know the depth of nodes, length of the ordering
        if depth_of_vars1 == 0:             #if no nodes, return 1
            return 1


        Y = vars1.pop(0)      #extract the first value from topologically sorted

        if Y.name in evidence.keys():   #if existing in the evidence, just read out probability and recursive go deeper with this removed from ordered list
            evidence_copy = evidence.copy()
            return Y.probability(evidence_copy[Y.name], evidence_copy)*self._enumerate_all(vars1, evidence_copy)
        else:       #if not exist in evidince
            summed_probs = 0    #to remember the sum
            for i in range(Y.no_states): #for each state
                evidencey = evidence.copy() 
                evidencey[Y.name] = i       #adds the variable with values to evidence
                summed_probs += Y.probability(i, evidencey)*self._enumerate_all(vars1, evidencey) #Sums up the probabilities
            return summed_probs #returns the total prob
        

    def query(self, var, evidence={}):
        """
        Wrapper around "_enumeration_ask" that returns a
        Tabular variable instead of a vector
        """
        q = self._enumeration_ask(var, evidence).reshape(-1, 1) #-1 means numpy will figure out the value. -1,1 gives array of arrays with 1 value [[1], [2], [3], ..] 1 equals column
        #reshape(-1,1) gir x kolonner med 1 elem i hver
        #reshape(1,-1) gir én rad med x elementer
        return Variable(var, self.bayesian_network.variables[var].no_states, q)


def problem3c():
    d1 = Variable('A', 2, [[0.8], [0.2]])
    d2 = Variable('B', 2, [[0.5, 0.2],
                           [0.5, 0.8]],
                  parents=['A'],
                  no_parent_states=[2])
    d3 = Variable('C', 2, [[0.1, 0.3],
                           [0.9, 0.7]],
                  parents=['B'],
                  no_parent_states=[2])
    d4 = Variable('D', 2, [[0.6, 0.8],
                           [0.4, 0.2]],
                  parents=['B'],
                  no_parent_states=[2])
    
    print(f"Probability distribution, P({d1.name})")
    print(d1)

    print(f"Probability distribution, P({d2.name} | {d1.name})")
    print(d2)

    print(f"Probability distribution, P({d3.name} | {d2.name})")
    print(d3)

    print(f"Probability distribution, P({d4.name} | {d2.name})")
    print(d4)
    
    bn = BayesianNetwork()

    bn.add_variable(d1)
    bn.add_variable(d2)
    bn.add_variable(d3)
    bn.add_variable(d4)
    bn.add_edge(d1, d2)
    bn.add_edge(d2, d3)
    bn.add_edge(d2, d4)
  
    inference = InferenceByEnumeration(bn) 
    posterior = inference.query('C', {'D': 1})
    
    print(f"Probability distribution, P({d3.name} | !{d4.name})")
    print(posterior)


def monty_hall():
    #pretty much copy the example provided 3c)...
    #creates the nodes for the structure
    d1 = Variable('CBG', 3, [[1/3],
                            [1/3],
                            [1/3]])
    d2 = Variable('P', 3, [[1/3],
                           [1/3],
                           [1/3]])
    d3 = Variable('OBH', 3, [[0., 0., 0., 0., 0.5, 1., 0., 1., 0.5],
                              [0.5, 0., 1., 0., 0., 0., 1., 0., 0.5],
                              [0.5, 1., 0., 1., 0.5, 0., 0., 0., 0.]],
                  parents=['CBG', 'P'],
                  no_parent_states=[3, 3])

    
    print(f"Probability distribution, P({d1.name})")
    print(d1)

    print(f"Probability distribution, P({d2.name})")
    print(d2)

    print(f"Probability distribution, P({d3.name} | {d2.name}, {d1.name})")
    print(d3)
    
    bn = BayesianNetwork()

    #add nodes and the edges
    bn.add_variable(d1)
    bn.add_variable(d2)
    bn.add_variable(d3)
    bn.add_edge(d1, d3)
    bn.add_edge(d2, d3)

    inference = InferenceByEnumeration(bn) 
    posterior = inference.query('P', {'CBG': 0, 'OBH': 2})  #the question asked
    
    print(f"Probability distribution, P({d2.name} | {d1.name} = 1 {d3.name} = 3)")
    print(posterior)



if __name__ == '__main__':
    #problem3c()
    monty_hall()