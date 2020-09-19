import numpy as np
from Layer import Layer

class Flatten(Layer) : 
    
    def __init__(self):
        super().__init__()

    
    def flattening(self,input_feature_map):
        return np.array(input_feature_map).flatten().tolist()
    
    def call(self, input):
        return self.flattening(input)


# matrix = [[[1,2,3],[2,3,4]],[[1,2,1],[5,4,2]]]
# flat = Flatten()
# result = flat.flattening(matrix)
# print(result)