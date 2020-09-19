import functools
from Layer import Layer
import cv2

class Sequential:
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

from Convolution import Convolution
# from Pooling import Pooling
from Detector import Detector
from Flatten import Flatten
from Dense import 
import numpy as np

s = Sequential()

# s.add(Pooling(filter_size=2, stride_size=1, mode="max"))
# s.add(Pooling(filter_size=2, stride_size=2, mode="max"))

s.add(Convolution(num_filter =  1,padding_size= 0,stride_size= 1, input_size = (10,10,3),filter_size = (4,3)))
s.add(Convolution(num_filter =  1,padding_size= 0,stride_size= 1,filter_size = (5,5)))
s.add(Flatten())
s.add(Dense(2,"relu"))
convo = Convolution(num_filter =  1,padding_size= 0,stride_size= 1, input_size = (350,350,3),filter_size = (4,3))
matrix_img = cv2.imread('test/cats/cat.0.jpg')
input_layer = list()
input_layer.append(convo.get_red_matrix(matrix_img))
input_layer.append(convo.get_green_matrix(matrix_img))
input_layer.append(convo.get_blue_matrix(matrix_img))
# input_layer = [[1,4,5,6,12,4,7,1,2,9],[1,4,5,6,12,4,7,1,2,9],[1,4,5,6,12,4,7,1,2,9],[1,4,5,6,12,4,7,1,2,9],[1,4,5,6,12,4,7,1,2,9],[1,4,5,6,12,4,7,1,2,9],[1,4,5,6,12,4,7,1,2,9],[1,4,5,6,12,4,7,1,2,9],[1,4,5,6,12,4,7,1,2,9],[1,4,5,6,12,4,7,1,2,9],[1,4,5,6,12,4,7,1,2,9]]
# feature_map = convo.convolution(input_layer)
# print(":::")
# print(feature_map)
# img = Image.fromarray(feature_map[0])
# img.show()

# mat = np.arange(25).reshape(5, 5)
# print(mat)
output = s.forward_prop(input_layer)

# train_ds = image_dataset_from_directory(
#     directory='test/',
#     labels='inferred',
#     label_mode='int',
#     batch_size=40,
#     image_size=(150,150))

# for images,labels in train_ds.take(1):
#     for i in tange(len(images)): 
#         print(labels[i].numpy())
#         list_images.append(images[i].numpy()*1/255)