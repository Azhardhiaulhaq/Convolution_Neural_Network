import random 
import numpy as np
from Convolution import Convolution
from Layer import Layer
from Flatten import Flatten

class Dense(Layer) :
    def __init__(self,num_unit,activation="relu"):
        super().__init__()
        self.num_unit = num_unit
        self.activation = activation
        self.bias = 1
        self.weights = None
        self.inputs = None
        self.delta_weights = None

    def propagate(self, input_array) :
        result = list()
        for i in range(len(self.weights)) : 
            output = self.dot_product(self.weights[i],input_array)
            if(self.activation == "relu") : 
                output = self.relu(output)
            elif (self.activation == "sigmoid") :
                output = self.sigmoid(output)
            result.append(output)
        return result
    
    def relu(self,num):
        return max(0,num)

    def sigmoid(self,num):
        return 1/(1+np.exp(-num))
    
    def dot_product(self,weight_array,input_array):
        result = 0
        for i in range(len(weight_array)):
            result = result + weight_array[i] * input_array[i]
        return result
    
    def call(self, input):
        input.append(self.bias)
        self.inputs = input
        if self.weights is None : 
            self.weights = np.random.uniform(-1,1,size=(self.num_unit,len(self.inputs)))
        return self.propagate(self.inputs)

    def back_propagation(self,error,momentum):
        self.activation_derivative(error)
        if self.delta_weights is None:
            self.delta_weights = self.weights_derivative(error)
        else:
            self.delta_weights = self.delta_weights*momentum + self.weights_derivative(error)
        return self.input_derivative(error)

    def update(self, learning_rate):
        self.weights -= learning_rate*self.delta_weights
        self.delta_weights = None

    def activation_derivative(self, error):
        if(self.activation == "relu"):
            self.relu_derivative(error)
        else :
            self.sigmoid_derivative(error)
    
    def relu_derivative(self,error):
        for i in range(len(error)):
            if(error[i] < 0):
                error[i] = 0
    
    def sigmoid_derivative(self,error):
        for i in range(len(error)):
            error[i] = error[i]*(1-error[i])

    def weights_derivative(self,error):
        result = list()
        for i in range(len(error)):
            weigth = list()
            for j in range(len(self.inputs)):
                weigth.append(error[i]*self.inputs[j])
            result.append(weigth)
        return np.array(result)
    
    def input_derivative(self,error):
        result = list()
        for i in range(len(self.weights[0])):
            result.append(np.dot(error,self.weights[:,i]))
        return np.array(result[:-1])


