#! /usr/bin/env python3

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Detector import Detector
from Pooling import Pooling
from Flatten import Flatten
# from MyCNN import MyCNN
import random
import cv2
from Layer import Layer


class Convolution(Layer):
    def __init__(self,
                 num_filter,
                 padding_size=0,
                 stride_size=1,
                 input_size=(),
                 filter_size=()):

        super().__init__()
        self.num_filter = num_filter
        self.input_size = input_size
        self.filter_size = filter_size
        self.padding_size = padding_size
        self.stride_size = stride_size
        self.bias = 0
        self.filters = []
        self.delta_weights = None
        if (input_size):
            self.filters = np.random.uniform(-1, 1, size=(
                self.num_filter, self.input_size[2], self.filter_size[0], self.filter_size[1]))

    def convolution(self, input_layer, filters, type='valid'):
        feature_map_row = int((
            (len(input_layer[0]) - len(filters[0][0])) / self.stride_size) + 1)
        feature_map_column = int((
            (len(input_layer[0][0]) - len(filters[0][0][0])) / self.stride_size) + 1)
        if type == 'valid':
            feature_map = np.zeros(self.filters.shape)
        elif type == 'forward':
            feature_map = np.zeros(
                (len(filters), feature_map_row, feature_map_column))
        else: #type == 'full']
            feature_map = np.zeros((
                len(self.filters), self.input_size[2],self.input_size[0], self.input_size[1])
            )
        for i in range(len(filters)):
            if(type == 'valid' or type=='forward'):
                feature_map[i] = self.count_feature_map(
                    input_layer, filters[i], feature_map_size=feature_map.shape[-2:], type=type)
            elif (type == 'full'):
                feature_map[i] = self.count_feature_map(input_layer[i], filters[i], feature_map_size=feature_map.shape[-2:], type='full')
        # print(feature_map.shape)
        return feature_map

    def resize_matrix(self, layer_matrix):
        layer = []
        for matrix in layer_matrix:
            layer.append(self.add_padding(
                cv2.resize(np.float32(matrix), dsize=(self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_CUBIC), self.padding_size))
        return layer

    def add_padding(self, matrix, padding_size):
        return np.pad(matrix, [padding_size, padding_size])

    def count_feature_map(self, input_layer, filter, feature_map_size=(), type='valid'):
        feature_map = np.zeros((feature_map_size[0], feature_map_size[1]))
        kernel_feature_map = np.zeros(
            (len(filter), feature_map_size[0], feature_map_size[1]))
        for i in range(len(kernel_feature_map)):
            for j in range(feature_map_size[0]):
                for k in range(feature_map_size[1]):
                    initial_row = j + self.stride_size - 1
                    initial_column = k + self.stride_size - 1
                    if(type == 'valid' or type =='forward'):
                        kernel_feature_map[i][j][k] = self.dot_product(input_layer[i][initial_row:initial_row+len(
                            filter[i]), initial_column:initial_column+len(filter[i][0])], filter[i])
                    elif(type == 'full'):
                        kernel_feature_map[i][j][k] = self.dot_product(input_layer[initial_row:initial_row+len(
                            filter[i]), initial_column:initial_column+len(filter[i][0])], filter[i])
        if type == 'forward':
            for kernel_map in kernel_feature_map:
                feature_map = feature_map + kernel_map
        else:
            return kernel_feature_map
        return feature_map

    def dot_product(self, input, filter):
        result = 0
        for i in range(len(input)):
            for j in range(len(input[0])):
                result = result + input[i][j]*filter[i][j]
        return result 
    

    def call(self, input):
        if(len(self.filters) == 0):
            self.filters = np.random.uniform(-1, 1, size=(
                self.num_filter, len(input), self.filter_size[0], self.filter_size[1]))
            self.input_size = (len(input[0][0]), len(input[0]), len(input))
        self.input = self.resize_matrix(input)
        self.bias = np.random.uniform(-1,1,size=(len(self.filters)))
        result = self.convolution(self.input, self.filters, type='forward')
        for i in range(len(result)):
            result[i] += self.bias[i]
        return result

    def back_propagation(self, error, momentum):
        if self.delta_weights is None:
            self.delta_weights = self.weights_derivative(error)
            self.delta_bias = np.zeros(self.bias.shape)
        else:
            self.delta_weights = self.weights_derivative(
                error) + momentum*self.delta_weights
            d_bias = np.zeros(self.bias.shape)
            for i in range(len(error)):
                d_bias[i] = error[i].sum()
            self.delta_bias = d_bias + momentum*self.delta_bias
        return self.input_derivative(error)

    def weights_derivative(self, error):
        list_error = list()
        error = np.asarray(error)
        for i in range(len(error)):
            list_error.append(error[i].reshape(
                1, error[i].shape[0], error[i].shape[1]))
        result = self.convolution(self.input, list_error)
        return np.array(result)

    def update(self, learning_rate):
        assert self.filters.shape == self.delta_weights.shape
        self.filters -= learning_rate *self.delta_weights
        self.bias -= learning_rate * self.delta_bias
        # self.filters = np.array([[channel - learning_rate*delta_weight for channel in filters]
        #                          for (filters, delta_weight) in zip(self.filters, self.delta_weights)])
        self.delta_weights = None
        self.delta_bias = 0

    def input_derivative(self, error):
        padded_error = list()
        for i in range(len(error)):
            padded_error.append(self.add_padding(error[i], 1))
        error = self.convolution(padded_error, self.filters, type='full')
        result = np.zeros(error.shape[1:])

        for filters in error:
            result += filters
        return result
