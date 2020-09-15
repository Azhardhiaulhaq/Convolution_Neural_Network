import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random


class Convolution:
    def __init__(self, 
    input_size, 
    filter_size,
    num_filter,
    padding_size, 
    stride_size):

        super().__init__()
        self.num_filter = num_filter
        self.input_size = input_size
        self.filter_size = filter_size
        self.padding_size = padding_size
        self.stride_size = stride_size
        self.bias = 0

    def get_red_matrix(self, image):
        result = image.copy()
        result[:, :, 1] = 0
        result[:, :, 2] = 0
        return result

    def get_green_matrix(self, image):
        result = image.copy()
        result[:, :, 0] = 0
        result[:, :, 2] = 0
        return result

    def get_blue_matrix(self, image):
        result = image.copy()
        result[:, :, 0] = 0
        result[:, :, 1] = 0
        return result

    def convolution(self, input_layer):

        self.filters = np.random.choice([-1,0,1],size = (self.num_filter,len(input_layer),self.filter_size,self.filter_size))
        
        input_layer = self.resize_matrix(input_layer)
        feature_map_size = (
            (len(input_layer[0]) - len(self.filters[0][0]) ) / self.stride_size) + 1
        if(not feature_map_size.is_integer() or not feature_map_size > 0):
            print("Feature Map Size is Not Integer")
        else:
            feature_map = self.forward_propagation(int(feature_map_size),input_layer,self.filters)
            return feature_map

    def resize_matrix(self, layer_matrix):
        layer = []
        for matrix in layer_matrix:
            layer.append(self.add_padding(
                np.resize(matrix, (self.input_size, self.input_size))))

        return layer

    def add_padding(self, matrix):
        return np.pad(matrix, [self.padding_size, self.padding_size])


    def forward_propagation(self,feature_map_size,input_layer,filters) :
        feature_map = np.zeros((len(filters),feature_map_size,feature_map_size))
        for i in range(len(filters)):
            feature_map[i] = self.count_feature_map(feature_map_size,input_layer,filters[i])
        return feature_map
        
    def count_feature_map(self,feature_map_size,input_layer,filter):
        feature_map = np.zeros((feature_map_size,feature_map_size))
        kernel_feature_map = np.zeros((len(filter),feature_map_size,feature_map_size))
        for i in range(len(kernel_feature_map)):
            for j in range(feature_map_size): 
                for k in range(feature_map_size):
                    initial_row = j + self.stride_size - 1
                    initial_column = k + self.stride_size - 1
                    kernel_feature_map[i][j][k] = self.dot_product(input_layer[i][initial_row:initial_row+len(filter[i]),initial_column:initial_column+len(filter[i])],filter[i])
        for kernel_map in kernel_feature_map : 
            feature_map = feature_map + kernel_map
        return feature_map
                    
    def dot_product(self,input,filter):
        result = 0
        for i in range(len(input)):
            for j in range(len(input[0])):
                result = result + input[i][j]*filter[i][j]
        return result + self.bias





# convo = Convolution(input_size = 300, filter_size = 2,num_filter =  3,padding_size= 0,stride_size= 1)
# img = Image.open('cats/cat.0.jpg')
# matrix_img = np.array(img)
# input_layer = list()
# # input_layer.append([[252,251,246,207,90],[250,242,236,144,41],[252,244,228,102,43],[250,243,214,59,52],[248,243,201,44,54]])
# # print(input_layer)
# input_layer.append(convo.get_red_matrix(matrix_img))
# input_layer.append(convo.get_green_matrix(matrix_img))
# input_layer.append(convo.get_blue_matrix(matrix_img))
# feature_map = convo.convolution(input_layer)
# print(":::")
# print(feature_map)
