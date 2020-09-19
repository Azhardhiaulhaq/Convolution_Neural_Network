import numpy as np
import math
from Layer import Layer

class Pooling(Layer):
    def __init__(self, 
    filter_size, 
    stride_size,
    mode):

        super().__init__()
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.mode = mode
    
    def count_feature_map_size(self, input):
        return ((len(input) - self.filter_size) / self.stride_size) + 1
    
    def get_func_mode(self):
        if self.mode == "mean":
            return lambda input: np.mean(input)
        elif self.mode == "max":
            return lambda input: np.max(input)

    def get_sub_mat(self, input, i, j):
        list_row = [self.stride_size*i + n for n in range(self.filter_size)]
        list_col = [self.stride_size*j + n for n in range(self.filter_size)]
        ixgrid = np.ix_(list_row,list_col)
        return input[ixgrid]

    def pooling_mat(self, input, feature_map_size, func_mode):
        feature_map = np.zeros((feature_map_size,feature_map_size))
        for i in range (feature_map_size):
            for j in range (feature_map_size):
                feature_map[i][j] = func_mode(self.get_sub_mat(input,i,j))
        return feature_map

    def pooling_opt(self, input_layer):
        feature_maps = []
        feature_map_size = self.count_feature_map_size(input_layer[0])
        func_mode = self.get_func_mode()
        if not feature_map_size.is_integer():
            feature_map_size = int(math.floor(feature_map_size))
        else:
            feature_map_size = int(feature_map_size)
        for layer in input_layer:
            feature_maps.append(self.pooling_mat(
                layer, 
                feature_map_size,
                func_mode))
        return feature_maps
    
    def call(self, input):
        return self.pooling_opt(input)

# layer = Layer()
# pool = Pooling(filter_size=3, stride_size=1, mode="max")
# func_mode = pool.get_func_mode()


# mat = np.arange(25).reshape(5, 5)
# # print(pool.count_feature_map_size(mat))
# print(mat)
# # print(pool.get_sub_mat(mat, 0, 0))
# # print(func_mode(pool.get_sub_mat(mat, 0,0)))
# print(pool.pooling([mat]))
# print(pool([mat]))