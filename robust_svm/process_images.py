from os.path import join,isfile,exists,isdir
import numpy as np
import glob
import cv2
from settings import hog3_feature_path
import settings
def get_successors(pixd,xlim,ylim):
    """
    
    :param pixd: 
    :param xlim: 
    :param ylim: 
    :return: 
    """



def grow_region(image):
    """
    
    :param image: 
    :return: 
    """
def hole_fill(image):
    """
    
    :param image: 
    :return: 
    """
def get_seeds(pixel,xlim,ylim,image):
    """
    
    :param pixel: 
    :param xlim: 
    :param ylim: 
    :param image: 
    :return: 
    """
def highlight_invariant_threashold(image):
    """
    
    :param image: 
    :return: 
    """
def image_to_feature_vector(image):
    """
    
    :param image: 
    :return: 
    """
def get_hog_features(data_class_id,image_file_name):
    """

    :param data_set_name:
    :param image_file_name:
    :return:
    """
    hog3_feature_path = "C:/Users/ch.srivamsi priyanka/Documents/GitHub/traffic_sign_svm/data/training_data/Features_HOG/HOG_3"
    image_feature_filename = image_file_name.replace(".ppm",".txt")
    hog_feature_path = join(hog3_feature_path,data_class_id,image_feature_filename)
    with open(hog_feature_path, 'r') as input_file:
        lines_of_file = [line for line in input_file]
    return np.array(lines_of_file)







if __name__ == "__main__":
    # Test the function read_image_annotations
    data_class_id='00000'
    image_file_name = '00000_00000.ppm'
    hog_feature_data= get_hog_features(data_class_id,image_file_name)
    print(hog_feature_data)
