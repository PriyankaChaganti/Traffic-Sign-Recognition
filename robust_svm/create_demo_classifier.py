from collections import defaultdict
from os.path import join
import pickle
from sklearn import svm
from time import time

from robust_svm.classifier import MultiClassClassifier
from robust_svm.read_images import make_dataset, get_all_datasets
from robust_svm import settings
from robust_svm.svm_utils import test_classifier

########################################################################################
# Data Processing
########################################################################################
# Compile a list of datasets that would be used for training and testing the classifier
data_set_list = ['00013', '00015']

# Set the type of image features that would be used for training ex: hog_1, hog_3,
# custom_features etc.
feature_path = settings.custom_features_2_path

# Set the path where the training data and test data would be stored as pickled objects
training_data_pickle_path = join(settings.dumps_folder, "demo_training_data.p")
test_data_pickle_path = join(settings.dumps_folder, "demo_test_data.p")

# Process the training images and load the training data
print('Loading training data')
training_data = make_dataset(settings.training_data_folder, feature_path, data_set_list)
pickle.dump(training_data, open(training_data_pickle_path, "wb"))
# Comment the above two lines and uncomment the following line to avoid re-processing images
# training_data = pickle.load(open(training_data_pickle_path, "rb"))

print('Loading test data')
test_data = make_dataset(settings.test_data_folder, feature_path, data_set_list)
pickle.dump(test_data, open(test_data_pickle_path, "wb"))
# Comment the above two lines and uncomment the following line to avoid re-processing images
# test_data = pickle.load(open(test_data_pickle_path, "rb"))


###########################################################################################
# Build Classifier
###########################################################################################
multi_svm_classifier_pickle_path = join(settings.dumps_folder, 'demo_multi_svm_classifier.p')
labels = []
features = []
for each_row in training_data.data:
    labels.append(each_row['class_id'])
    features.append(each_row['feature_vector'])

multi_svm_classifier = svm.SVC(gamma='scale')
multi_svm_classifier.fit(features, labels)

pickle.dump(multi_svm_classifier, open(multi_svm_classifier_pickle_path, "wb"))
# Comment the above two lines and uncomment the following line to avoid re-building classifier
# multi_svm_classifier = pickle.load(open(multi_svm_classifier_pickle_path, "rb"))


##########################################################################################
# Test Classifier
###########################################################################################

expected_labels = []
test_features = []
for each_row in test_data.data:
    expected_labels.append(each_row['class_id'])
    test_features.append(each_row['feature_vector'])

predicted_labels = multi_svm_classifier.predict(test_features)
predicted_labels = predicted_labels.tolist()

accuracy_results = defaultdict(lambda: {'right': 0, 'wrong': 0, 'total': 0})
for el, pl in zip(expected_labels, predicted_labels):
    if el == pl:
        accuracy_results[el]['right'] += 1
    else:
        accuracy_results[el]['wrong'] += 1
    accuracy_results[el]['total'] += 1

print("Accuracy Results:")
for class_id, results in accuracy_results.items():
    results['accuracy_percentage'] = format((results['right'] / results['total'])*100,'.2f')
    print('class id: {}'.format(class_id), results)
