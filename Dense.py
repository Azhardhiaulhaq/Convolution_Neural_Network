import random 
import numpy as np
from Convolution import Convolution
from Layer import Layer
from Flatten import Flatten

class Dense(Layer) :
    def __init__(self,num_unit,activation):
        super().__init__()
        self.num_unit = num_unit
        self.activation = activation
        self.bias = 1

    def propagate(self, input_matrix) :
        f = Flatten()
        input_array = f.flattening(input_matrix) 
        result = list()
        input_array.append(self.bias)
        weights = np.random.rand(self.num_unit,len(input_array))
        for i in range(len(weights)) : 
            output = self.dot_product(weights[i],input_array)
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
        return self.propagate(input)

# arr = [1,-2,3,-4,5]
# result = list()
# dens = Dense(2,"relu")
# dens2 = Dense(1,"sigmoid")
# print(dens2.propagate(arr))