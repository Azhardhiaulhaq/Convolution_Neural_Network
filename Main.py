import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Convolution import Convolution
from Detector import Detector
from Pooling import Pooling
from Flatten import Flatten
from MyCNN import MyCNN
from Dense import Dense
import cv2

class Main:
    def __init__(self):
        super.__init__()

    def predict(self, output):
        pass

    def get_red_matrix(self, image):
        return image[:,:,2]

    def get_green_matrix(self, image):
        return image[:,:,1]

    def get_blue_matrix(self, image):
        return image[:,:,0]

    if __name__ == "__main__":
        list_images = []
        list_labels = []
        list_predictions = []

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory='test/',
            labels='inferred',
            label_mode='int',
            batch_size=40,
            image_size=(300,300))

        for images,labels in train_ds.take(1):
            for i in range(len(images)): 
                list_labels.append(labels[i].numpy())
                list_images.append(images[i].numpy())

        model = MyCNN()
        model.add(Convolution(num_filter =  16, input_size = (150,150,3),filter_size = (3,3)))
        model.add(Detector())
        model.add(Pooling(filter_size=(2,2), stride_size=1, mode="max"))
        model.add(Convolution(num_filter =  32,filter_size = (3,3)))
        model.add(Detector())
        model.add(Pooling(filter_size=(2,2), stride_size=1, mode="max"))
        model.add(Flatten())
        model.add(Dense(128,"relu"))
        model.add(Dense(64,"relu"))
        model.add(Dense(1,"sigmoid"))

        for image in list_images :
            list_prediction.append(model.forward_prop(image))
        
        print('| Predictions    | Labels    |')
        for i in range(len(list_labels)):
            print('| ' + str(list_predictions[i] + '\t| ' + str(list_labels[i] + '\t|')))
            



