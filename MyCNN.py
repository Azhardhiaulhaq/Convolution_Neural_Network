import functools
from Layer import Layer
import cv2

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
            error = layer.back_propagation(error,momentum)
        return error

    # Update All Weight on all Layer
    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def printlayer(self):
        for layer in self.layers:
            print(layer.__dict__)

    # TODO
    # fit trains model
    def fit(self, X, y, epoch, learning_rate, momentum):
        for i in range(epoch):
            print('Iteraesi : ',i)
            output = self.forward_prop(X)
            # self.printlayer()
            self.backward_prop(output, momentum)
            # self.printlayer()
            self.update(learning_rate)
            # self.printlayer()