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

    def back_propagation(self,error):
        list_error = np.asarray(error)
        return list_error.reshape(self.input_shape)

# flat = Flatten()
# input = np.asarray([[[0,76,64],[109,0,10],[118,71,67]],[[0,0,66],[0,102,0],[0,0,0]]])
# out = flat.call(input)
# print(out)
# out2 = flat.back_propagation(out)
# print(out2)