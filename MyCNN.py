import functools
from Layer import Layer
import cv2
import jsonpickle
import numpy as np

class MyCNN:
    def __init__(self, 
    layers=None):
        super().__init__()
        self.layers = []
        if layers:
            if not isinstance(layers, (list, tuple)):
                layers = [layers]
            for layer in layers:
                self.add(layer)

    # add adds new layer into layers list
    def add(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError("Not Layer Type")
        else:
            self.layers.append(layer)

    # pop deletes last layer
    def pop(self):
        if self.layers:
            self.layers.pop()
        else:
            raise ValueError("Empty layers")
    
    # forward_prop
    def forward_prop(self, input_data):
        for layer in self.layers:
            output = layer(input_data)
            input_data = output
        return output

    # TODO
    def predict(self,X):
        pass

    # backward_prop
    def backward_prop(self, error, momentum):
        for layer in reversed(self.layers):
            error = layer.back_propagation(error, momentum)
        return error

    # Update All Weight on all Layer
    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def printlayer(self):
        for layer in self.layers:
            print(layer.__dict__)
    
    def create_mini_batches(self, X, y, batch_size):
        X_mini_batches = [] 
        y_mini_batches = []
        data = list(zip(X, y))
        np.random.shuffle(data)
        X, y = zip(*data)

        for i in range(len(X)):
            if i * batch_size >= len(X):
                break 
            X_mini_batches.append(X[i * batch_size:(i + 1)*batch_size]) 
            y_mini_batches.append(y[i * batch_size:(i + 1)*batch_size])
        
        return X_mini_batches, y_mini_batches
        
    # fit trains model
    def fit(self, X, y, epoch, learning_rate, momentum, batch_size):
        X_mini_batches, y_mini_batches = create_mini_batches(X, y, batch_size)
        data_batches = list(zip(X_mini_batches, y_mini_batches))

        for i in range(epoch):
            for data in data_batches:
                for data_X, data_y in zip(*data):
                    print("forward prop")
                    output = self.forward_prop(X[j])
                    print("backward prop")
                    self.backward_prop(output, momentum)
                self.update(learning_rate)
            