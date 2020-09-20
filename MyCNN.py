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

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError("Not Layer Type")
        else:
            self.layers.append(layer)

    def pop(self):
        if self.layers:
            self.layers.pop()
        else:
            raise ValueError("Empty layers")

    def forward_prop(self, input_data):
        for layer in self.layers:
            output = layer(input_data)
            print(output)
            input_data = output
        return output

    # TODO 
    def backward_prop(self):
        pass

    # TODO
    def fit(self):
        pass