from Convolution import Convolution
from Pooling import Pooling
from Detector import Detector
from Sequential import Sequential
from Dense import Dense
import cv2

class Main:
    def __init__(self):
        super.__init__()

s = Sequential()
convolution = Convolution(input_size = 30, filter_size = 3, num_filter =  1, padding_size= 1, stride_size= 1)
detector = Detector()
pooling = Pooling(filter_size=(2,2), stride_size=1, mode="max")
dense = Dense(num_unit=2, activation="sigmoid")
s.add(convolution)
s.add(detector)
s.add(pooling)
s.add(dense)

matrix_img = cv2.imread('cats/cat.0.jpg')
input_layer = list()
input_layer.append(convolution.get_red_matrix(matrix_img))
input_layer.append(convolution.get_green_matrix(matrix_img))
input_layer.append(convolution.get_blue_matrix(matrix_img))

print(s.forward_prop(input_layer))