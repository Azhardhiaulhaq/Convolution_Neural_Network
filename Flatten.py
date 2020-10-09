import numpy as np
from Layer import Layer

class Flatten(Layer) : 
    
    def __init__(self):
        super().__init__()

    
    def flattening(self,input_feature_map):
        return np.array(input_feature_map).flatten().tolist()
    
    def call(self, input):
        self.input_shape = input.shape
        return self.flattening(input)

    def back_propagation(self,error,momentum):
        list_error = np.asarray(error)
        return list_error.reshape(self.input_shape)
