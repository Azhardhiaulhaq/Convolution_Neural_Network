#! /usr/bin/env python3

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import cv2


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
        return image[:,:,2]

    def get_green_matrix(self, image):
        return image[:,:,1]

    def get_blue_matrix(self, image):
        return image[:,:,0]

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
                cv2.resize(matrix,dsize=(self.input_size, self.input_size),interpolation=cv2.INTER_CUBIC)))


        return layer

    def add_padding(self, matrix):
        return np.pad(matrix, [self.padding_size, self.padding_size])


    def forward_propagation(self,feature_map_size,input_layer,filters) :
        feature_map = np.zeros((len(filters),feature_map_size,feature_map_size),dtype=np.uint8)
        for i in range(len(filters)):
            feature_map[i] = self.count_feature_map(feature_map_size,input_layer,filters[i])
        return feature_map
        
    def count_feature_map(self,feature_map_size,input_layer,filter):
        feature_map = np.zeros((feature_map_size,feature_map_size),dtype=np.uint8)
        kernel_feature_map = np.zeros((len(filter),feature_map_size,feature_map_size),dtype=np.uint8)
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

    def relu(self, input):
        for i in range(len(input)):
            for j in range(len(input[0])):
                if input[i][j] < 0:
                   input[i][j] = 0

def relu(input):
    for i in range(len(input)):
        for j in range(len(input[0])):
            if input[i][j] < 0:
                input[i][j] = 0  

class Pooling:
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

    def pooling(self, input_layer):
        feature_maps = []
        feature_map_size = self.count_feature_map_size(input_layer[0])
        func_mode = self.get_func_mode()
        if not feature_map_size.is_integer():
            return None
        else:
            feature_map_size = int(feature_map_size)
        for layer in input_layer:
            feature_maps.append(self.pooling_mat(
                layer, 
                feature_map_size,
                func_mode))
        return feature_maps
    
# mat = [[1,1,-1],[2,-2,2],[-3,3,3]]
# relu(mat)
# print(mat)
# # convo = Convolution(input_size = 350, filter_size = 3,num_filter =  1,padding_size= 0,stride_size= 1)
# matrix_img = cv2.imread('cats/cat.0.jpg')
# input_layer = list()
# input_layer.append(convo.get_red_matrix(matrix_img))
# input_layer.append(convo.get_green_matrix(matrix_img))
# input_layer.append(convo.get_blue_matrix(matrix_img))
# feature_map = convo.convolution(input_layer)
# print(":::")
# print(feature_map)
# img = Image.fromarray(feature_map[0])
# img.show()

# img = cv2.imread('cats/cat.0.jpg')
# print(image)
# cv2.imshow("Image",image)
# cv2.waitKey()
# img = Image.open('cats/cat.0.jpg')
# matrix_img = np.array(img)

# pool = Pooling(filter_size=5, stride_size=1, mode="max")
# func_mode = pool.get_func_mode()


# mat = np.arange(25).reshape(5, 5)
# # print(pool.count_feature_map_size(mat))
# print(mat)
# # print(pool.get_sub_mat(mat, 0, 0))
# # print(func_mode(pool.get_sub_mat(mat, 0,0)))
# print(pool.pooling([mat]))