from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

################################ Oppgave 1 #######################################
#1a) Formulate information given as hidden markov model, provide complete probability tables for the model
#There are a drawing in the handwritten part of the pdf. 

'''
https://www.nature.com/articles/s41586-020-2649-2

Xt = fish nearby at day t

P(Xt) =             0.5 =   P(fish in lake) = 0.5
P(Xt | Xt-1) =      0.8 =   P(fish nearby at day t | fish nearby on day t - 1)
P(Xt | not Xt-1) =  0.3 =   P(fish nearby at day t | not fish nearby on day t - 1)
P(Et | Xt) =        0.75 =  P(birds nearby at day t | fish nearby at day t)
P(Et | not Xt) =    0.2 =   P(birds nearby at day t | not fish nearby at day t)

e1 = {birds nearby}
e2 = {birds nearby}
e3 = {no birds nearby}
e4 = {birds nearby}
e5 = {no birds nearby}
e6 = {birds nearby}

P(X0) = <0.5, 0.5>

Xt-1    |P(Xt)
--------|-----
True    |0.8
False   |0.3

Xt      |P(Et)
--------|-----
True    |0.75
False   |0.25

'''

#1b) Compute P(Xt|e1:t), t=1,2,..,6. 
# What kind of operation is this (filtering, prediction, smoothing, likelihood of the evidence, or
# most likely sequence)? Describe in words what kond of information this operation provides us. 
'''This is filtering. This is in practice a update of the prediction by new evidence. We therefore compute Xt with the evidence from e1 to e6.'''

#1c) Compute P(Xt|e1:6), t=7,8,..,30.
#What kind of operation is this? Describe in words what kond of information this operation provides us. What happens to distribution in Eq 2) as t increases? 
'''This is a prediction. We only have evidence for the values e1 to e6, but we try to estimate the values from 7 to 30 based on the probability of the outcome of the value before. When t increases, the probability will get closer and closer and in the end the same as the stationary state, and in the end converge towards this value.'''

#1d) Compute P(Xt|e1:6), t=0,1,..,5
#What kind of operation is this? Describe in words what kind of information this operation provides us. 
'''This is smoothing, we try to estimate the earlier state is true based on the evidence we now have. This is important for learning. Given the evidence we have now, what were the info for earlier states.'''

#1e) Compute argMax(x1,.., xt-1) P(x1, x2, .., xt-1, Xt|e1:t), t=1,2,..6.
#What kind of operation is this? Describe in words what kind of information this operation provides us.
'''This is a operation for finding the MLS, or the most likely sequence. This computes the the most likely sequence to xt given the most likely path to state xt.'''

class problem_1_construct:
    """
    constructor for problems
    """
    def __init__(self):
        self.transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])    # Transition model, P(Xt | Xt-1)
        self.observation_matrix = np.array([[0.75, 0.0], [0.0, 0.2]])   #sensory model. Constant for filtering
        
        self.prior = np.array([0.5,0.5])               #Prior probability, initial state probabilities
        self.evidence = np.array([1, 1, 0, 1, 0, 1])    #indicates the evidence, if birds nearby or not  
    
    
class manipulator:
    """
    Methods for operations on HMM, among others filtering, prediction, etc. 
    Transpose matrix: ndarray.T or ndarray.transpose()

    """
    def __init__(self, problem):
        self.problem_to_compute = problem


    def forward(self, prior, depth, t=-1, storage = [], prediction = False):
        """
        f 1:t+1 = alpha*Ot+1*Transpose(T)*f 1:t
        eks.
        prior
        f 1:0 = P(X0) = <0.5,0.5>
        t = 0
        f 1:1 = alpha*O1*Transpose(T)*f 1:0 = alpha*[[0.75, 0], [0, 0.2]]*[[0.8, 0.2],[0.3, 0.7]]*[0.5, 0.5]
        function for forwarding. Used to compute filtering.
        """

        if t == -1:     #Check if first iteration, change equal depth if true
            t = depth
        
        T = self.problem_to_compute.transition_matrix.copy().T  #transition matrix Transformed
        O = self.problem_to_compute.observation_matrix.copy()   #Observarion matrix 
        if self.problem_to_compute.evidence[depth-t] == 0:      #Check if the evidence is true or false
            O = np.array([[1.0, 0.0], [0.0, 1.0]]) - O          #if false, change to obs-table valid for false
        if t == 0:  #if at the bottom, last day, t is counting downwards
            storage.append(self.normalize(np.matmul(np.matmul(O, T), prior))[0]) #appends last prob to storage
            return self.normalize(np.matmul(np.matmul(O, T), prior)), storage   #returns the normalized last prob
        else:
            O = self.normalize(np.matmul(np.matmul(O, T), prior)) #Is not in the end of the recursive path
            #print(t)
            storage.append(O[0])    #Appends the value to the storage, used for plotting and printing probabilities
            prior = self.forward(O, depth, t-1, storage)    #prior is the probability for last element. Not entirely correct name...
            return prior

    def forward_pred(self, prior, depth, t=-1, storage = [], prediction = False):
        """
        f 1:t+1 = alpha*Ot+1*Transpose(T)*f 1:t
        eks.
        prior
        f 1:0 = P(X0) = <0.5,0.5>
        t = 0
        f 1:1 = alpha*O1*Transpose(T)*f 1:0 = alpha*[[0.75, 0], [0, 0.2]]*[[0.8, 0.2],[0.3, 0.7]]*[0.5, 0.5]
        Used to calculate prediction, although it is not entirely correct... 
        """

        if t == -1:     #Check if first iteration, change equal depth if true
            t = depth
        
        T = self.problem_to_compute.transition_matrix.copy().T  #transition matrix Transformed
        O = self.problem_to_compute.observation_matrix.copy()   #Observarion matrix 
        
        if self.problem_to_compute.evidence[depth-t] == 0:      #Check if the evidence is true or false
            O = np.array([[1.0, 0.0], [0.0, 1.0]]) - O          #if false, change to obs-table valid for false
        

        if t == 0:  #if at the bottom, last day, t is counting downwards
            storage.append(self.normalize(np.matmul(np.matmul(O, T), prior))[0]) #appends last prob to storage
            return self.normalize(np.matmul(np.matmul(O, T), prior)), storage   #returns the normalized last prob
        elif t > 24:    #We have observations for values from 30 to 24(the 6 first days, counting backwards)
            O = self.normalize(np.matmul(np.matmul(O, T), prior)) 
            storage.append(O[0])
            prior = self.forward_pred(O, depth, t-1, storage)
            return prior
        else: #If deeper than 6, prediciton function start. This is the part that probabily are wrong, not sure what to send further down in the recursive function... If figured this out, the prediction shoud probably converge against statonary state, wich i doubt is what is printed; <0.923, 0.077>
            const_prior = prior.copy()
            O = self.normalize(np.matmul(np.matmul(O, T), const_prior))
            storage.append(O[0])
            prior = self.forward_pred(O, depth, t-1, storage)
            return prior

    def backward(self):
        """
        Not implemented
        """
    
    def forward_backward(self):
        """
        Not implemented

        """
    
    def viterbi(self):
        """ 
        Not implemented        
        """
    
    def normalize(self, vector):
        """
        Used to normalize vectors.
        """
        return vector/np.sum(vector)


def problem1b(): 
    """
    Code for solving task 1b)
    """
    prob1 = problem_1_construct()   #Construct the problem
    task_calc = manipulator(prob1)  #creates the manipulator
    result = task_calc.forward(prob1.prior, 5) #Calulates the probabilities by forward, filtering.
    evidence_table = prob1.evidence.copy()  #Takes a copy of evidence table
    new_evidence = []   #crates new to name values accordingly to true/false
    for i in range(len(evidence_table)):
        if evidence_table[i] == 1:
            new_evidence.append("True")
        else:
            new_evidence.append("False")

    #A try to print out the soulution nicw...
    print("The probabilities by filtering with evidence from day 1 to 6:\n")
    for i in range(len(result[1])):
        print("The probability at day ", i+1, " with evidence; Birds nearby:", new_evidence[i], " are:\t [true, false] = [", "%.5f"%result[1][i], "%.5f"%(1-result[1][i]), "]")
    
    #Code for plotting the result. Yes, in norwegian... 
    t0 = 1          #Start
    tf = 6       #Slutt
    N = 6        #Antall datapunkter
    t = np.linspace(t0, tf, N) #evalueringspunkter, N tall mellom t0 og tf
    
    plt.plot(t, result[1], "g")      #plotter inn funksjonen i rød
    plt.axis([0, 6, 0, 1])
    plt.xlabel("t = days")             #navngir x-aksen
    plt.grid(True)              #legger til grid
    plt.ylabel("Probability")  #navngir y-aksen
    plt.legend(["P( X(t) | e(1:t))"])    #Navngir inne i grid, legend(["graph1, graph2, .."])
    plt.show()
    

def problem1c():
    """
    The code for solving task 1c)
    Is exactly the same as 1b) only that we have datapoints up to 30 because thes is our domain where we should try to predict values. 
    As said, this is not entirely correct becouse of the (i think) wrong convergence... What should I send further in the recurion when passing day 6??
    """
    prob1 = problem_1_construct()   
    task_calc = manipulator(prob1)
    evidence_table = list(prob1.evidence.copy())
    for i in range(5,30):
        evidence_table.append(-1)
    prob1.evidence = np.array(evidence_table)

    
    result = task_calc.forward_pred(prob1.prior, 30)
    
    new_evidence = []
    for i in range(len(evidence_table)):
        if evidence_table[i] == 1:
            new_evidence.append("True")
        elif evidence_table[i] == -1:
            new_evidence.append("Prediction")
        else:
            new_evidence.append("False")

    print("The probabilities by filtering with evidence from day 1 to 6:\n")
    for i in range(len(result[1])-1):
        print("The probability at day ", i+1, " with evidence ", new_evidence[i], " are:\t [true, false] = [", "%.5f"%result[1][i], "%.5f"%(1-result[1][i]), "]")

    t0 = 1          #Start
    tf = 30       #Slutt
    N = 31        #Antall datapunkter
    t = np.linspace(t0, tf, N) #evalueringspunkter, N tall mellom t0 og tf

    plt.plot(t, result[1], "g")      #plotter inn funksjonen i rød
    plt.axis([0, 30, 0, 1])
    plt.xlabel("t = days")             #navngir x-aksen
    plt.grid(True)              #legger til grid
    plt.ylabel("Probability")  #navngir y-aksen
    plt.legend(["P( X(t) | e(1:6) )"])    #Navngir inne i grid, legend(["graph1, graph2, .."])
    plt.show()

def main():
    problem1b()
    problem1c()

main()


