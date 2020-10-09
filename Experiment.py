import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from Convolution import Convolution
from Detector import Detector
from Pooling import Pooling
from Flatten import Flatten
from MyCNN import MyCNN
from Dense import Dense
import cv2, os
from PIL import Image
from keras.preprocessing.image import array_to_img
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import accuracy_score, confusion_matrix

class Experiment:
    def __init__(self):
        super.__init__()

    # TODO
    def predict(self, output):
        pass
        
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
        images = np.array([(cv2.resize(cv2.cvtColor((cv2.imread(img, 1)), cv2.COLOR_BGR2RGB), (150,150))) for img in file_path])

        return images, class_label

    def schema_split(model, list_images, list_labels):

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        
        for train_index, test_index in sss.split(list_images, list_labels):
            training_data, testing_data = list_images[train_index], list_images[test_index]
            training_label, testing_label = list_labels[train_index], list_labels[test_index]

        model.fit(training_data, training_label, epoch=10, learning_rate=0.5, momentum=0.5, batch_size=2) # TRAIN MODEL
        pred_label = model.predict(testing_data) # PREDICT DATA
            
        accuracy = accuracy_score(testing_label, pred_label)
        print("Accuracy : ", accuracy)

        print("Confussion Matrix : ")
        tn, fp, fn, tp = confusion_matrix(testing_label, pred_label).ravel()
        print("| tp |", tp)
        print("| fp |", fp)
        print("| tn |", tn)
        print("| fn |", fn)

    def schema_cross_validation(model, k, list_images, list_labels):
        
        kf = KFold(n_splits=k)

        for train_index, test_index in kf.split(input_data):
            training_data, testing_data = list_images[train_index], list_images[test_index]
            training_label, testing_label = list_labels[train_index], list_labels[test_index]
           
            model.fit(training_data, training_label, epoch=10, learning_rate=0.5, momentum=0.5, batch_size=2) # TRAIN MODEL
            pred_label = model.predict(testing_data) # PREDICT DATA

            accuracy = accuracy_score(testing_label, pred_label)
            print("Accuracy : ", accuracy)

    
    if __name__ == "__main__":
        
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        
        dataset_path = "/content/drive/My Drive/Colab Notebooks/CNN/train"
        train_data, train_label = read_image(dataset_path)

        dataset_path = "/content/drive/My Drive/Colab Notebooks/CNN/test"
        test_data, test_label = read_image(dataset_path)

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

        schema_split(model, train_data, test_label)
        schema_cross_validation(model, train_data, test_label)