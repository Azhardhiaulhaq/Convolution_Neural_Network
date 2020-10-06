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

    def back_propagation(self,error,target_weight):
        self.activation_derivative(error)
        delta_weight = list()
        if(target_weight == True):
            delta_weigth = self.input_derivative(error)
        elif (target_weight == False): 
            delta_weight = self.weights_derivative(error)
        return delta_weight

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

    def input_derivative(self,error):
        result = list()
        for i in range(len(error)):
            weigth = list()
            for j in range(len(self.inputs)):
                weigth.append(error[i]*self.inputs[j])
            result.append(weigth)
        return result
    
    def weights_derivative(self,error):
        result = list()
        for i in range(len(self.weights[0])):
            result.append(np.dot(error,self.weights[:,i]))
        return result

# dense = Dense(10,"relu")
# input = [10,0]
# error = [2,1,0,10,-3,5,10,7,-4,-8]
# output = dense.call(input)
# dense.back_propagation(error,False)
