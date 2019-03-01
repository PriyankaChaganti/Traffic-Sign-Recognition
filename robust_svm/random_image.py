"""
The file contains three sets of differently configured HOG features and a custom algorithm which preprocess the images.
The accuracy is later compared for each algorithm.
 """
import cv2
import numpy
from os.path import join
import pickle

from robust_svm import settings
from robust_svm.image_processing_utils import *
from robust_svm.svm_utils import determine_image_class


def custom_feature_transformation(image_file_path):
    """
    The function reads the image and processes the image to extract the feature_vector of the image
    :param image_file_path:The path which leads to the image.
    :return:row
    """
    image = cv2.imread(image_file_path)
    thresholded_image = highlight_invariant_threshold(image)
    filled_image = hole_fill(thresholded_image)
    region_grown_image = grow_region(filled_image)
    image_array = image_to_array(region_grown_image)
    feature_vector = numpy.array(image_array)
    row = {'feature_vector': feature_vector}
    return row


def hog_1_transformation():
    pass


def hog_2_transformation():
    pass


def hog_3_transformation():
    pass


# Folder containing the random image
random_images_path = join(settings.project_path, "data", "random_images")
image_file_path = join(random_images_path, "15.ppm")

# Load the multi-class classifier from dumps.
multi_svm_classifier_pickle_path = join(settings.dumps_folder, 'multi_svm_classifier.p')
multi_svm_classifier = pickle.load(open(multi_svm_classifier_pickle_path, "rb"))

# Transform the image into the feature that was used while building the classifier
row = custom_feature_transformation(image_file_path)

# Classify the image using multi-class classifier
all_labels = multi_svm_classifier.get_all_classifier_labels(row)
class_id = determine_image_class(all_labels)
print('The Multi-class SVM classifer has classified the image as {0}'.format(class_id))
