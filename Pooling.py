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

    # count_feature_map_size returns feature_map_size shape
    def count_feature_map_size(self, input_shape):
        
        feature_map_size_row =  math.floor(((input_shape[0] - self.filter_size[0]) / self.stride_size) + 1)
        feature_map_size_col =  math.floor(((input_shape[1] - self.filter_size[1]) / self.stride_size) + 1)

        return feature_map_size_row, feature_map_size_col
    
    # get_func_mode returns function used for pooling
    def get_func_mode(self):
        if self.mode == "mean":
            return lambda input: np.mean(input)
        elif self.mode == "max":
            return lambda input: np.max(input)
        else:
            raise ValueError("Method {} unsupported".format(self.mode))
    
    # get_sub_mat returns receptive field
    def get_sub_mat(self, input, i, j):
        filter_size_row, filter_size_col = self.filter_size
        list_row = [self.stride_size*i + n for n in range(filter_size_row)]
        list_col = [self.stride_size*j + n for n in range(filter_size_col)]
        ixgrid = np.ix_(list_row,list_col)
        return input[ixgrid]

    # count_feature_map returns feature map from 1 channel
    def count_feature_map(self, input, feature_map_size, func_mode):
        feature_map_size_row, feature_map_size_col = feature_map_size
        feature_map = np.zeros((feature_map_size_row,feature_map_size_col))

        for i in range (feature_map_size_row):
            for j in range (feature_map_size_col):
                feature_map[i][j] = func_mode(self.get_sub_mat(input,i,j))
        return feature_map

    # pooling returns feature maps after pooling
    def pooling(self, input_layer):
        feature_map_size = self.count_feature_map_size(input_layer[0].shape)
        func_mode = self.get_func_mode()
        result = self.forward_propagation(feature_map_size,input_layer,func_mode)
        self.result_pooling = result
        return result
    
    # forward_propagation counts all feature maps
    def forward_propagation(self, feature_map_size,input_layer,func_mode):
        feature_maps = np.zeros((len(input_layer),feature_map_size[0],feature_map_size[1]))
        for i, layer in enumerate(input_layer):
            feature_maps[i] = self.count_feature_map(layer, feature_map_size,func_mode)
        return feature_maps
    
    def call(self, input):
        self.input = input
        self.input_shape = input.shape
        return self.pooling(input)

    def back_propagation(self,error,momentum):
        result = np.zeros(self.input_shape)
        for i in range(len(error)):
            for j in range(len(error[0])):
                for k in range(len(error[0][0])):
                    position = np.where(self.input == self.result_pooling[i][j][k])
                    result[position[0][0]][position[1][0]][position[2][0]] = error[i][j][k]
        return result


# input = np.asarray([[[0,76,64],[109,0,10],[118,71,67]],[[0,0,66],[0,102,0],[0,0,0]]])
# print(np.where(input == 109))
# pool = Pooling(filter_size=(3,3), stride_size=1, mode="max")
# out = pool.call(input)

# result = pool.back_propagation([[[0.079]],[[0.239]]])
# print(result)