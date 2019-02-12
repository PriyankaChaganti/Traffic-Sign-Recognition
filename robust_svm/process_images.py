from os.path import join
import numpy as np
import glob
from settings import hog3_feature_path
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
    hog_feature_path = join(data_class_id,image_file_name)
    files = sorted(glob.glob(hog_feature_path + '/*.txt'))
    arrays = []
    for f in files:
        arrays.append(np.load(f))
    hog_feature_data = np.concatenate(arrays)
    return hog_feature_data









if __name__ == "__main__":
    # Test the function read_image_annotations
    data_set_name='../data/training_data/Features_HOG/HOG_3'
    image_file_name = '00000'
    hog_feature_data= get_hog_features(data_set_name,image_file_name)
    print(hog_feature_data)
