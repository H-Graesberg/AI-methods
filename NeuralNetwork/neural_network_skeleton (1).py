# Use Python 3.8 or newer (https://www.python.org/downloads/)
import unittest
# Remember to install numpy (https://numpy.org/install/)!
import numpy as np
import pickle
import os
import random
#np.random.seed(10)


class NeuralNetwork:
    """Implement/make changes to places in the code that contains #TODO."""

    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        # --- PLEASE READ --
        # Use the parameters below to train your feed-forward neural network.

        # Number of hidden units if hidden_layer = True.
        self.hidden_units = 25

        # This parameter is called the step size, also known as the learning rate (lr).
        # See 18.6.1 in AIMA 3rd edition (page 719).
        # This is the value of α on Line 25 in Figure 18.24.
        self.lr = (1e-3)*2 

        ##################### QUESTION! ########################### 
        # With the default lr and weights in the range [-0.5, 0.5) the accuracy varied between 0.7 and 0.97, fail in 1/10 times. When doubling the lr and reducing the weigths to [-0.25, 0.25) i have constantly accuracy over 0.9. Why??? One co-student made the algorithm purely with matrices, why is his NN so stable with a lower value for lr? Does my loops affect this? I tried with a seed and the code reproduced the same result over and over. Can it be due to overfitting; unlucky with high weights results in memorization in some cases? And other cases, when lucky with weights, generalize well? So now when init weights are low and lr higher the risk of overfitting is reduced?:) Appro. runningtime: 30-50sek

        # Line 6 in Figure 18.24 says "repeat".
        # This is the number of times we are going to repeat. This is often known as epochs.
        self.epochs = 400

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        # TODO: Make necessary changes here. For example, assigning the arguments "input_dim" and "hidden_layer" to
        # variables and so forth.
        
        #Create nodes for input layer
        #just the length is important.. not sure wich method is fastest? ones, zeros, random.rand, etc. Think that empty(shape) is fastest?
        self.input_nodes = np.empty(input_dim)

        #Create output node, value do not matter
        self.output_node = 0

        #creates the bool for hidden layer as an attribute
        self.hidden_layer = hidden_layer  

        #In this implementation, the value of biases is equal for whole layer with activation value of 1. Therefore these do not affect the implementation, but the thresholds for activation can be adjusted by changing these values. Remark the counting is FROM input nodes TOWARDS output, can be a bit confusing that bias 1 is for outputnode when no hidden layer, and for hidden layer when it is present. Should have counted from output towards input.
        self.bias_1 = 1 
        
        #The delta (error) for output node
        self.delta_output = 0

        #this initialization is just useful for perceptrons with x number of inputs OR NN with 1 hidden layer
        if (hidden_layer):

            #creates deltas for hidden layer
            self.delta_hidden_layer = np.empty(self.hidden_units) 

            #Weights from input layar to hidden units; shape: (30, 25), values: [-0.25, 0.25). Random chosen, most important to have relatively small weights to reduce risk of overfitting. With 0.5 the accuracy were less stable
            self.weights_1 = (np.random.rand(input_dim, self.hidden_units) - 0.5)/2
            
            #The weight for the bias for the hidden layer
            self.bias_1_weight = (np.random.rand(self.hidden_units) - 0.5)/2

            #create nodes for hidden layer, the values does not matter, are changed later anyway. Just the shape.
            #self.hidden_nodes = np.random.rand(self.hidden_units)
            self.hidden_nodes = np.empty(self.hidden_units)

            #Weights from hidden layer to output node; shape: (25,1), values: [-0.25, 0.25)
            self.weights_2 = (np.random.rand(self.hidden_units, 1) - 0.5)/2

            #bias for output node, can be changed to adjust the threshold for activation
            self.bias_2 = 1

            #The weights from bias_2 to output node, just a single value
            self.bias_2_weight = (np.random.random() - 0.5)/2

        else: #Perceptron, directly from input to output node (perceptron)
            
            #weights from input layer to output node, shape (30,1)
            self.weights_1 = (np.random.rand(input_dim, 1) - 0.5)/2 #get the weights closer to 0, reduce risk of overfitting and extreme weights
            
            #bias weight for the output node
            self.bias_1_weight = (np.random.random() - 0.5)/2




    def load_data(self, file_path: str = os.path.join(os.getcwd(), 'data_breast_cancer.p')) -> None:
        """
        Do not change anything in this method.

        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data['x_train'], data['y_train']
            self.x_test, self.y_test = data['x_test'], data['y_test']

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network.
        Small random numbers were set for weights in initialization"""
        # TODO: Implement the back-propagation algorithm outlined in Figure 18.24 (page 734) in AIMA 3rd edition.
        # Only parts of the algorithm need to be implemented since we are only going for one hidden layer. 
        # 
        # Update python!! 
        # Experienced that when I first coded in python 3.7 the algorithm were over 0.8 each time, but when I updated to 3.9 the accuracy became less accurate. Had to double lr and reduce the init weights to [-0.25, 0.25)

        # Line 6 in Figure 18.24 says "repeat".
        # We are going to repeat self.epochs times as written in the __init()__ method.

        # Line 27 in Figure 18.24 says "return network". Here you do not need to return anything as we are coding
        # the neural network as a class
        
        #Number of epochs (rounds of training)
        for i in range(self.epochs):
            
            #For each training example
            for example_num in range(len(self.y_train)):

                #Updates input nodes to have value from this example
                self.input_nodes = self.x_train[example_num]
                
                
                #If contain hidden layer
                if (self.hidden_layer): 

                    #Sets the activation value for hidden nodes, could probably be done without loop (matrices), try if have time
                    for k in range(len(self.hidden_nodes)):
                        self.hidden_nodes[k] = self.set_activation(self.input_nodes, self.weights_1[:,k], k) 
                    
                    #Sets the activation value for output node
                    self.output_node = self.set_activation(self.hidden_nodes, self.weights_2) 

                    #Computes delta(error) of output node. Derived sigma is hardcoded, should be changed to its own function for more flexibility for reuse of code.
                    self.delta_output = (self.y_train[example_num] - self.output_node)*self.output_node*(1-self.output_node) 

                    #computes the deltas for the hidden nodes
                    for k in range(len(self.hidden_nodes)):
                        self.delta_hidden_layer[k] = self.hidden_nodes[k]*(1-self.hidden_nodes[k])*(self.weights_2[k]*self.delta_output)
                    
                    #Updates weights between input and hidden. Remark the layers are counting from input
                    self.weights_1 += self.input_nodes.reshape(30,1)*self.lr*self.delta_hidden_layer 
                    
                    #updates bias weight for hidden layer
                    self.bias_1_weight += self.lr*self.delta_hidden_layer*self.bias_1
                    
                    #updates weights between hidden and output
                    self.weights_2 += self.hidden_nodes.reshape(25,1)*self.lr*self.delta_output

                    #updates bias weight for output
                    self.bias_2_weight += self.lr*self.delta_output*self.bias_2

                else:
                    #only one perceptron, updates output node by setting its activation
                    self.output_node = self.set_activation(self.input_nodes, self.weights_1)

                    #computes the delta for the output node
                    self.delta_output = (self.y_train[example_num] - self.output_node)*self.output_node*(1-self.output_node)
                    
                    #updates weights from input to perceptron(output)
                    self.weights_1 += self.input_nodes.reshape(30,1)*self.lr*self.delta_output

                    #Updates weight for bias to output
                    self.bias_1_weight += self.lr*self.delta_output*self.bias_1

    def sigmoid(self, x):
        """
        The sigmoid function, used to compute acivation value
        """
        return 1/(1+np.exp(-x))
    
    def set_activation(self, input_value, weights, hidden = -1):
        """
        Computes the activation value for a node, could been better if the counting were from the output and not the input (same code for updating the output with self.bias_2_weight for both)
        """
        #if hidden layer present
        if self.hidden_layer:
            if hidden >=0: #If a node from the hidden layer (to use right weights, improved by changing counting direction)
                return self.sigmoid(np.dot(input_value, weights) + self.bias_1_weight[hidden]*self.bias_1)
            
            else: #if the output value
                return self.sigmoid(np.dot(input_value, weights) + self.bias_2_weight*self.bias_2)
        #No hidden layer present, only a perceptron
        else:
            return self.sigmoid(np.dot(input_value, weights) + self.bias_1_weight*self.bias_1)
        
    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        #if hidden layer present
        if (self.hidden_layer):
            #Sets the activation value for hidden nodes, x is input values and self.weights_1[:,k] is the column (correct weights) to given node
            for k in range(len(self.hidden_nodes)):
                self.hidden_nodes[k] = self.set_activation(x, self.weights_1[:,k])
                    
            #Sets the activation value for output node
            self.output_node = self.set_activation(self.hidden_nodes, self.weights_2) 
        
        else: #No hidden layer, just predict the output node
            self.output_node = self.set_activation(x, self.weights_1)
        
        return self.output_node
       


class TestAssignment5(unittest.TestCase):
    """
    Do not change anything in this test class.

    --- PLEASE READ ---
    Run the unit tests to test the correctness of your implementation.
    This unit test is provided for you to check whether this delivery adheres to the assignment instructions
    and whether the implementation is likely correct or not.
    If the unit tests fail, then the assignment is not correctly implemented.
    """

    def setUp(self) -> None:
        self.threshold = 0.8
        self.nn_class = NeuralNetwork
        self.n_features = 30

    def get_accuracy(self) -> float:
        """Calculate classification accuracy on the test dataset."""
        self.network.load_data()
        self.network.train()

        n = len(self.network.y_test)
        correct = 0
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = self.network.predict(self.network.x_test[i])
            # Sanity check of the prediction
            #print("\nprediction: ", pred)
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            correct += self.network.y_test[i] == round(float(pred))
        return round(correct / n, 3)

    #@unittest.skip("will test later")
    def test_perceptron(self) -> None:
        """Run this method to see if Part 1 is implemented correctly."""

        self.network = self.nn_class(self.n_features, False)
        accuracy = self.get_accuracy()
        print("\nAccuracy for perceptron: ", accuracy) #Thought it would be nice to print the accuracy:)
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')
    
    def test_one_hidden(self) -> None:
        """Run this method to see if Part 2 is implemented correctly."""

        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        print("\nAccuracy for multilayer: ", accuracy) #Here too
        #print("input - hidden: self.network.weights_1", "\n\nhidden - output: \n", self.network.weights_2)
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')


if __name__ == '__main__':
    unittest.main()

