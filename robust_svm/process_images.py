from os.path import join
import numpy as np
import glob
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
def get_hog_features(data_set_name,image_file_name):
    """

    :param data_set_name:
    :param image_file_name:
    :return:
    """
    hog_feature_path = join(data_set_name,image_file_name)
    files = sorted(glob.glob(hog_feature_path + '/*.npy'))
    arrays = []
    for f in files:
        arrays.append(np.load(f))
    hog_feature_data = np.concatenate(arrays)
    return hog_feature_data









if __name__ == "__main__":
    # Test the function read_image_annotations
    data_set_name='../data/training_data/Features_HOG/'
    image_file_name = 'HOG_3'
    hog_feature_data= get_hog_features(data_set_name,image_file_name)
    print(hog_feature_data)
