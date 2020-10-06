import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from layer import Convolution, Detector, Pooling, Flatten, Dense
from MyCNN import MyCNN
import cv2
from PIL import Image
from tensorflow.keras.preprocessing.image import array_to_img

class Main:
    def __init__(self):
        super.__init__()
    # TODO
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
        treshold = 0.5

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory='test/',
            labels='inferred',
            label_mode='int',
            batch_size=40,
            image_size=(300,300))

        for images,labels in train_ds.take(1):
            for i in range(len(images)): 
                list_labels.append(labels[i].numpy())
                list_images.append(images[i].numpy()*1/255)


        model = MyCNN()
        model.add(Convolution(num_filter =  4, input_size = (150,150,3),filter_size = (3,3)))
        model.add(Detector())
        model.add(Pooling(filter_size=(2,2), stride_size=1, mode="max"))
        model.add(Convolution(num_filter =  8,filter_size = (3,3)))
        model.add(Detector())
        model.add(Pooling(filter_size=(2,2), stride_size=1, mode="max"))
        model.add(Flatten())
        model.add(Dense(256,"relu"))
        model.add(Dense(1,"sigmoid"))

        for i in range(len(list_images)) :
            print('Iterasi ke - ' + str(i+1))
            predict = model.forward_prop(list_images[i])
            if (predict[0] > treshold):
                list_predictions.append(1)
            else :
                list_predictions.append(0)
            
        
        print('| Predictions    | Labels    |')
        for i in range(len(list_labels)):
            print('| ' + str(list_predictions[i]) + '\t| ' + str(list_labels[i]) + '\t|')
            



