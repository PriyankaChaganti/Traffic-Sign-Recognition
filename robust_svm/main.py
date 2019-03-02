from os.path import join
import pickle
from time import time

from robust_svm.classifier import MultiClassClassifier
from robust_svm.read_images import make_dataset
from robust_svm import settings
from robust_svm.svm_utils import test_classifier

########################################################################################
# Data Processing
########################################################################################
# Compile a list of datasets that would be used for training and testing the classifier
data_set_list = ['00017', '00019']

# Set the type of image features that would be used for training ex: hog_1, hog_3,
# custom_features etc.
feature_path = settings.hog3_path

# Set the path where the training data and test data would be stored as pickled objects
training_data_pickle_path = join(settings.dumps_folder, "training_data.p")
test_data_pickle_path = join(settings.dumps_folder, "test_data.p")

# Process the training images and load the training data
print('Loading training data')
training_data = make_dataset(settings.training_data_folder, feature_path, data_set_list)
pickle.dump(training_data, open(training_data_pickle_path, "wb"))
# Comment the above two lines and uncomment the following line to avoid re-processing images
#training_data = pickle.load(open(training_data_pickle_path, "rb"))

# Process the test images and load the test data
print('Loading testing data')
test_data = make_dataset(settings.test_data_folder, feature_path, data_set_list)
pickle.dump(test_data, open(test_data_pickle_path, "wb"))
# Comment the above two lines and uncomment the following line to avoid re-processing images
#test_data = pickle.load(open(test_data_pickle_path, "rb"))

###########################################################################################
# Build Classifier
###########################################################################################
# Set the parameters for Multi-class Classifier
epochs = 100
svm_params = {
    'r0': 0.1,
    'C': 0.7,
    'kernel_type': 'linear'
}
multi_svm_classifier_pickle_path = join(settings.dumps_folder, 'multi_svm_classifier.p')

# Build a classifier using the training_data
print('Building the classifier using epochs={0} and svm_params={1}'.format(
    epochs, svm_params), end=' ')
multi_svm_classifier = MultiClassClassifier(training_data, epochs, svm_params)
pickle.dump(multi_svm_classifier, open(multi_svm_classifier_pickle_path, "wb"))
# Comment the above two lines and uncomment the following line to avoid re-building classifier
#multi_svm_classifier = pickle.load(open(multi_svm_classifier_pickle_path, "rb"))


##########################################################################################
# Test Classifier
###########################################################################################
accuracy_results = test_classifier(multi_svm_classifier,test_data)
print("Accuracy Results:")
for class_id, results in accuracy_results.items():
    results['accuracy_percentage'] = format((results['right'] / results['total'])*100,'.2f')
    print('class id: {}'.format(class_id), results)
