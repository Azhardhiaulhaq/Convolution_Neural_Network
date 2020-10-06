import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from layer import Convolution, Detector, Pooling, Flatten, Dense
from MyCNN import MyCNN
import cv2, os
from PIL import Image
from tensorflow.keras.preprocessing.image import array_to_img
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold

class Experiment:
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
        
    def read_image(dataset_path):

        list_folder_per_class = os.listdir(dataset_path)
        list_folder_per_class = sorted(list_folder_per_class)
        file_path = []; class_label = np.ndarray(shape=(0)); class_dictionary = {}
        
        for i in range(len(list_folder_per_class)):
            class_folder_path = os.path.join(dataset_path, list_folder_per_class[i])
            list_image_name = os.listdir(class_folder_path)
            list_image_name = sorted(list_image_name)
            temp_file_path = [os.path.join(class_folder_path, j) for j in list_image_name]
            file_path += temp_file_path
            temp_class_label = np.full((len(list_image_name)),np.int16(i))
            class_label = np.concatenate((class_label, temp_class_label), axis=0)
            class_dictionary[str(i)] = list_folder_per_class[i]
        
        file_path, class_label, class_dictionary = np.asarray(file_path), np.array(class_label), class_dictionary
        images = np.array([(cv2.resize(cv2.cvtColor((cv2.imread(img, 1)), cv2.COLOR_BGR2RGB), (300,300))) for img in file_path])

        return images, class_label
    
    if __name__ == "__main__":
        list_images = []
        list_labels = []
        list_predictions = []
        treshold = 0.5

        dataset_path = "./train"
        train_data, train_label = read_image(dataset_path)
        for data in train_data:
            list_images.append(data)
        for label in train_label:
            list_labels.append(label) 

        dataset_path = "./test"
        test_data, test_label = read_image(dataset_path)
        for data in test_data:
            list_images.append(data)
        for label in test_label:
            list_labels.append(label)

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

        list_images = np.array(list_images)
        list_labels = np.array(list_labels)

        # Experiment 1
        # Skema split 90% train data, 10% test data
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        for train_index, test_index in sss.split(list_images, list_labels):
            training_data, testing_data = list_images[train_index], list_images[test_index]
            training_label, testing_label = list_labels[train_index], list_labels[test_index]

        # Experiment 2
        # Skema 10-fold cross validation
        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(list_images):
            training_data, testing_data = list_images[train_index], list_images[test_index]
            training_label, testing_label = list_labels[train_index], list_labels[test_index]
            
        # for i in range(len(list_images)) :
        #     print('Iterasi ke - ' + str(i+1))
        #     predict = model.forward_prop(list_images[i])
        #     if (predict[0] > treshold):
        #         list_predictions.append(1)
        #     else :
        #         list_predictions.append(0)
            
        
        # print('| Predictions    | Labels    |')
        # for i in range(len(list_labels)):
        #     print('| ' + str(list_predictions[i]) + '\t| ' + str(list_labels[i]) + '\t|')
            



