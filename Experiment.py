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
        images = np.array([(cv2.resize(cv2.cvtColor((cv2.imread(img, 1)), cv2.COLOR_BGR2RGB), (150,150))) for img in file_path])

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
        model.add(Convolution(num_filter =  4, input_size = (150,150,3), filter_size = (3,3)))
        model.add(Detector())
        model.add(Pooling(filter_size=(2,2), stride_size=1, mode="max"))
        model.add(Convolution(num_filter =  8,filter_size = (3,3)))
        model.add(Detector())
        model.add(Pooling(filter_size=(2,2), stride_size=1, mode="max"))
        model.add(Flatten())
        model.add(Dense(256,"relu"))
        model.add(Dense(1,"sigmoid"))

        # MODEL MACHINE LEARNING
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (2,2), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.AveragePooling2D(2,2),
            tf.keras.layers.Conv2D(32, (2,2), activation='relu'),
            tf.keras.layers.AveragePooling2D(2,2), 
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
            tf.keras.layers.AveragePooling2D(2,2),
            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(512, activation='relu'), 
            tf.keras.layers.Dense(1, activation='sigmoid')  
        ])
        model.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy', metrics = ['accuracy'])

        list_images = np.array(list_images)
        list_labels = np.array(list_labels)

        # Experiment 1
        # Skema split 90% train data, 10% test data
        print("DATA SPLIT SCHEMA")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        for train_index, test_index in sss.split(list_images, list_labels):
            training_data, testing_data = list_images[train_index], list_images[test_index]
            training_label, testing_label = list_labels[train_index], list_labels[test_index]

        model.fit(training_data, training_label, epochs=10, verbose=2) # TRAIN MODEL
        pred_label = model.predict_classes(testing_data, batch_size=10) # PREDICT DATA
        # pred_label = []
        # i = 0
        # for data in testing_data:
        #     print("data ke-", i)
        #     predict = model.forward_prop(data)
        #     pred_label.append(predict)
        #     i = i + 1
            
        accuracy = accuracy_score(testing_label, pred_label)
        print("Accuracy : ", accuracy)

        print("Confussion Matrix : ")
        tn, fp, fn, tp = confusion_matrix(testing_label, pred_label).ravel()
        print("    | 1 | 0 |")
        print("| 1 |",tp,"|",fp)
        print("| 0 |",fn,"|",tn)

        # Experiment 2  
        # Skema 10-fold cross validation
        # print("10-FOLD CROSS VALIDATION SCHEMA")
        # kf = KFold(n_splits=10)
        # for train_index, test_index in kf.split(list_images):
        #     training_data, testing_data = list_images[train_index], list_images[test_index]
        #     training_label, testing_label = list_labels[train_index], list_labels[test_index]
           
        #     model.fit(training_data, training_label, epochs=5, verbose=2) # TRAIN MODEL
        #     pred_label = model.predict_classes(testing_data, batch_size=10) # PREDICT DATA

        #     accuracy = accuracy_score(testing_label, pred_label)
        #     print("Accuracy : ", accuracy)



