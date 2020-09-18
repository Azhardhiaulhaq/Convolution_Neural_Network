import numpy as np

class Flatten : 
    
    def __init__(self):
        super().__init__()

    
    def flattening(self,input_feature_map):
        return np.array(input_feature_map).flatten().tolist()


matrix = [[[1,2,3],[2,3,4]],[[1,2,1],[5,4,2]]]
flat = Flatten()
result = flat.flattening(matrix)
print(result)