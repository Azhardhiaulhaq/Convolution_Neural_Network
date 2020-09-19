#! /usr/bin/env python3

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import cv2
from Layer import Layer


class Convolution(Layer):
    def __init__ (self, 
    num_filter,
    padding_size, 
    stride_size,
    input_size = (), 
    filter_size = (),):

        super().__init__()
        self.num_filter = num_filter
        self.input_size = input_size
        self.filter_size = filter_size
        self.padding_size = padding_size
        self.stride_size = stride_size
        self.bias = 0
        self.filters = np.random.uniform(-1,1,size = (self.num_filter,self.input_size[2],self.filter_size[0],self.filter_size[1]))

    def get_red_matrix(self, image):
        return image[:,:,2]

    def get_green_matrix(self, image):
        return image[:,:,1]

    def get_blue_matrix(self, image):
        return image[:,:,0]

    def convolution(self, input_layer):
        input_layer = self.resize_matrix(input_layer)
        feature_map_row = int((
            (len(input_layer[0]) - len(self.filters[0][0]) ) / self.stride_size) + 1)
        feature_map_column = int((
            (len(input_layer[0][0]) - len(self.filters[0][0][0]) ) / self.stride_size) + 1)
        feature_map = self.forward_propagation(input_layer,self.filters,feature_map_size = (feature_map_row,feature_map_column))
        return feature_map

    def resize_matrix(self, layer_matrix):
        layer = []
        for matrix in layer_matrix:
            layer.append(self.add_padding(
                cv2.resize(matrix,dsize=(self.input_size[1], self.input_size[0]),interpolation=cv2.INTER_CUBIC)))


        return layer

    def add_padding(self, matrix):
        return np.pad(matrix, [self.padding_size, self.padding_size])


    def forward_propagation(self,input_layer,filters,feature_map_size = ()) :
        feature_map = np.zeros((len(filters),feature_map_size[0],feature_map_size[1]),dtype=np.uint8)
        for i in range(len(filters)):
            feature_map[i] = self.count_feature_map(input_layer,filters[i],feature_map_size = (feature_map_size[0],feature_map_size[1]))
        return feature_map
        
    def count_feature_map(self,input_layer,filter,feature_map_size=()):
        feature_map = np.zeros((feature_map_size[0],feature_map_size[1]),dtype=np.uint8)
        kernel_feature_map = np.zeros((len(filter),feature_map_size[0],feature_map_size[1]),dtype=np.uint8)
        for i in range(len(kernel_feature_map)):
            for j in range(feature_map_size[0]): 
                for k in range(feature_map_size[1]):
                    initial_row = j + self.stride_size - 1
                    initial_column = k + self.stride_size - 1
                    kernel_feature_map[i][j][k] = self.dot_product(input_layer[i][initial_row:initial_row+len(filter[i]),initial_column:initial_column+len(filter[i][0])],filter[i])
        for kernel_map in kernel_feature_map : 
            feature_map = feature_map + kernel_map
        return feature_map
                    
    def dot_product(self,input,filter):
        result = 0
        for i in range(len(input)):
            for j in range(len(input[0])):
                result = result + input[i][j]*filter[i][j]
        return result + self.bias
    
    def call(self, input):
        return self.convolution(input)
    
# convo = Convolution(num_filter =  1,padding_size= 0,stride_size= 1,input_size = (350,300,3), filter_size = (4,3))
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

