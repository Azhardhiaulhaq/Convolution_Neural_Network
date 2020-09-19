import numpy as np
from Layer import Layer

class Flatten(Layer) : 
    
    def __init__(self):
        super().__init__()

    
    def flattening(self,input_feature_map):
        return np.array(input_feature_map).flatten().tolist()
    
    def call(self, input):
        return self.flattening(input)


# matrix = [[[31,151,191],
#   [59,202 ,41],
#   [37 ,88,155],
#   [34,222,208]]]
# flat = Flatten()
# result = flat.flattening(matrix)
# print(result)