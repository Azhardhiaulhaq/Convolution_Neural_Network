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
from tensorflow.keras.preprocessing.image import array_to_img
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

        model.fit(training_data, training_label, epoch=3, learning_rate=0.5, momentum=0.5, batch_size=40) # TRAIN MODEL
        pred_label = model.predict(testing_data) # PREDICT DATA
            
        accuracy = accuracy_score(testing_label, pred_label)
        print("Accuracy : ", accuracy)

        print("Confussion Matrix : ")
        tn, fp, fn, tp = confusion_matrix(testing_label, pred_label).ravel()
        print("| tp |", tp)
        print("| fp |", fp)
        print("| tn |", tn)
        print("| fn |", fn)

    def schema_cross_validation(k, list_images, list_labels):
        
        kf = KFold(n_splits=k)

        best_model = create_model()
        best_accuracy = -999

        for train_index, test_index in kf.split(list_images):
            training_data, testing_data = list_images[train_index], list_images[test_index]
            training_label, testing_label = list_labels[train_index], list_labels[test_index]

            model = create_model()
           
            model.fit(training_data, training_label, epoch=1, learning_rate=0.5, momentum=0.5, batch_size=2) # TRAIN MODEL
            pred_label = model.predict(testing_data) # PREDICT DATA

            accuracy = accuracy_score(testing_label, pred_label)
            print("Accuracy : ", accuracy)
            
            if(accuracy > best_accuracy):
                best_model = model
        
        return best_model
    
    def create_model():
        model = MyCNN()
        model.add(Convolution(num_filter= 4, input_size=(150,150,3), filter_size=(3,3)))
        model.add(Detector())
        model.add(Pooling(filter_size=(2,2), stride_size=1, mode="max"))
        model.add(Flatten())
        model.add(Dense(256,"relu"))
        model.add(Dense(1,"sigmoid"))
        return model

    if __name__ == "__main__":
        
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        
        dataset_path = "train"
        train_data, train_label = read_image(dataset_path)

        dataset_path = "test"
        test_data, test_label = read_image(dataset_path)

        model_split = create_model()
        schema_split(model_split, train_data, train_label)
        model_split.save('Scheme_split.json')
        pred_label = model_split.predict(test_data)
        acc = accuracy_score(test_label,pred_label)
        print('Accuracy : ',acc)

        print("Confussion Matrix : ")
        tn, fp, fn, tp = confusion_matrix(testing_label, pred_label).ravel()
        print("| tp |", tp)
        print("| fp |", fp)
        print("| tn |", tn)
        print("| fn |", fn)

        model_crossval = schema_cross_validation(model_crossval, 10, train_data, train_label)
        model_crossval.save('schema_cross_validation_1_epoch.json') 
        pred_label = model_crossval.predict(test_data)
        acc = accuracy_score(test_label,pred_label)
        print('Accuracy : ',acc)

        print("Confussion Matrix : ")
        tn, fp, fn, tp = confusion_matrix(testing_label, pred_label).ravel()
        print("| tp |", tp)
        print("| fp |", fp)
        print("| tn |", tn)
        print("| fn |", fn)