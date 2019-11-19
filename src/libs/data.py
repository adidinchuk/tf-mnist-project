import heapq
from mlxtend.data import loadlocal_mnist
import numpy as np

class DataWrapper:        

    def __init__(self):
        self.training_inputs = None
        self.training_outputs = None
        self.testing_inputs = None
        self.testing_outputs = None
    
    def set_training_inputs(self, data):
        self.training_inputs = data

    def set_training_outputs(self, data):
        self.training_outputs = data

    def set_testing_inputs(self, data):
        self.testing_inputs = data

    def set_testing_outputs(self, data):
        self.testing_outputs = data
    
    def get_training_data(self):
        return self.training_inputs, self.training_outputs

    def get_testing_data(self):
        return self.testing_inputs, self.testing_outputs
    
    def debinarize(self, dataset):
        return [heapq.nlargest(1, range(len(x)), x.__getitem__) for x in dataset]

class MNISTProcessor:    

    def __init__(self, file_path, train_labels, train_images, test_labels, test_images):
        self.file_path = file_path
        self.train_labels = train_labels
        self.train_images = train_images
        self.test_labels = test_labels
        self.test_images = test_images
        self.data_wrapper = DataWrapper()
    
    def load_train(self, one_hot=True, normalize=True):
        print('Loading training data.')
        images, labels = loadlocal_mnist(images_path=self.file_path + self.train_images, labels_path=self.file_path + self.train_labels)
        if normalize:
            print('Normalizing images.')
            images = np.true_divide(images, 255)
        if one_hot:
            print('Converting labels to one-hot.')
            labels = self.transform_labels(labels)
        print('Image shape: ' + str(images[0].shape) + ' Label shape: ' + str(labels[0].shape))        
        self.data_wrapper.set_training_inputs(images)
        self.data_wrapper.set_training_outputs(labels)
        return self.data_wrapper

    def load_test(self, binarize=True, normalize=True):
        print('Loading test data.')
        images, labels = loadlocal_mnist(images_path=self.file_path + self.test_images, labels_path=self.file_path + self.test_labels)
        if normalize:
            print('Normalizing images.')
            images = np.true_divide(images, 255)
        if binarize:
            labels = self.transform_labels(labels)
            print('Binarizing labels.')
        print('Image shape: ' + str(images[0].shape) + ' Label shape: ' + str(labels[0].shape)) 
        self.data_wrapper.set_testing_inputs(images)
        self.data_wrapper.set_testing_outputs(labels)
        return self.data_wrapper

    #transforms labels from an integer value 0-9 to a binary array of length 10
    def transform_labels(self, labels):
        index_template = np.identity(10)
        return np.array([index_template[x] for x in labels])

    def debinarize(self, dataset):
        return [heapq.nlargest(1, range(len(x)), x.__getitem__) for x in dataset]
