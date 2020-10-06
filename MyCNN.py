import functools
from layer import Layer
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
    def backward_prop(self):
        pass

    # TODO
    def fit(self):
        pass
    
    # TODO
    def save(self,filename):
        pass

    # TODO
    def load(self, filename):
        pass