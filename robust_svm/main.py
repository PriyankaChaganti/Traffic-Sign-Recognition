from os.path import join

import pickle
from robust_svm.classifier import MultiClassClassifier
from robust_svm.read_images import make_dataset
from robust_svm import settings
from robust_svm.svm_utils import test_classifier

########################################################################################
# Data Processing
########################################################################################
# Compile a list of datasets that would be used for training and testing the classifier
data_set_list = ['00013', '00015']

# Set the type of image features that would be used for training ex: hog_1, hog_3,
# custom_features etc.
feature_path = settings.hog3_path

# Set the path where the training data and test data would be stored as pickled objects
training_data_pickle_path = join(settings.dumps_folder, "training_data.p")
test_data_pickle_path = join(settings.dumps_folder, "test_data.p")

# Process the training images and load the training data
#training_data = make_dataset(settings.training_data_folder, feature_path, data_set_list)
#pickle.dump(training_data, open(training_data_pickle_path, "wb"))
# Comment the above two lines and uncomment the following line to avoid re-processing images
training_data = pickle.load(open(training_data_pickle_path, "rb"))

# Process the test images and load the test data
#test_data = make_dataset(settings.test_data_folder, feature_path, data_set_list)
#pickle.dump(test_data, open(test_data_pickle_path, "wb"))
# Comment the above two lines and uncomment the following line to avoid re-processing images
test_data = pickle.load(open(test_data_pickle_path, "rb"))


###########################################################################################
# Build Classifier
###########################################################################################
# Set the parameters for Multi-class Classifier
epochs = 1
svm_params = {
    'r0': 1,
    'C': 1,
    'kernel_type': 'linear'
}
multi_svm_classifier_pickle_path = join(settings.dumps_folder, 'multi_svm_classifier.p')

# Build a classifier using the training_data
#multi_svm_classifier = MultiClassClassifier(training_data, epochs, svm_params)
#pickle.dump(multi_svm_classifier, open(multi_svm_classifier_pickle_path, "wb"))
# Comment the above two lines and uncomment the following line to avoid re-building classifier
multi_svm_classifier = pickle.load(open(multi_svm_classifier_pickle_path, "rb"))


##########################################################################################
# Test Classifier
###########################################################################################

label_sum = test_classifier(training_data,test_data)
