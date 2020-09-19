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
    # TODO Fix size in tupple
    def count_feature_map_size(self, input_shape):
        
        feature_map_size_row =  math.floor(((input_shape[0] - self.filter_size[0]) / self.stride_size) + 1)
        feature_map_size_col =  math.floor(((input_shape[1] - self.filter_size[1]) / self.stride_size) + 1)

        return feature_map_size_row, feature_map_size_col
    
    def get_func_mode(self):
        if self.mode == "mean":
            return lambda input: np.mean(input)
        elif self.mode == "max":
            return lambda input: np.max(input)
        else:
            raise ValueError("Method {} unsupported".format(self.mode))

    def get_sub_mat(self, input, i, j):
        filter_size_row, filter_size_col = self.filter_size
        list_row = [self.stride_size*i + n for n in range(filter_size_row)]
        list_col = [self.stride_size*j + n for n in range(filter_size_col)]
        ixgrid = np.ix_(list_row,list_col)
        return input[ixgrid]

    def count_feature_map(self, input, feature_map_size, func_mode):
        feature_map_size_row, feature_map_size_col = feature_map_size
        feature_map = np.zeros((feature_map_size_row,feature_map_size_col))

        for i in range (feature_map_size_row):
            for j in range (feature_map_size_col):
                feature_map[i][j] = func_mode(self.get_sub_mat(input,i,j))
        
        return feature_map

    def pooling(self, input_layer):
        feature_map_size = self.count_feature_map_size(input_layer[0].shape)
        func_mode = self.get_func_mode()
        return self.forward_propagation(feature_map_size,input_layer,func_mode)
    
    def forward_propagation(self, feature_map_size,input_layer,func_mode):
        feature_maps = np.zeros((len(input_layer),feature_map_size[0],feature_map_size[1]),dtype=np.uint8)
        for i, layer in enumerate(input_layer):
            feature_maps[i] = self.count_feature_map(layer, feature_map_size,func_mode)
        return feature_maps
    
    def call(self, input):
        return self.pooling(input)

# layer = Layer()
# pool = Pooling(filter_size=(3,2), stride_size=2, mode="max")
# func_mode = pool.get_func_mode()


# mat = np.arange(20).reshape(5, 4)
# # print(pool.count_feature_map_size(mat))
# print(mat)
# # print(pool.get_sub_mat(mat, 0, 0))
# # print(func_mode(pool.get_sub_mat(mat, 0,0)))
# print(pool.call([mat]))
# print(pool([mat]))