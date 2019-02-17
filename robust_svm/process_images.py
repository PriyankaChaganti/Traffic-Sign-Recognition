from os.path import join
import numpy as np
import cv2
from settings import hog3_folder

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

    hsvimage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    #blue
    treshLow1 = np.array([235, 0, 0], dtype=np.uint8)
    treshHigh1 = np.array([255, 255, 255], dtype=np.uint8)
    img1 = cv2.inRange(hsvimage, treshLow1, treshHigh1)

    treshLow1 = np.array([0, 40, 0], dtype=np.uint8)
    treshHigh1 = np.array([255, 255, 255], dtype=np.uint8)
    img2 = cv2.inRange(hsvimage, treshLow1, treshHigh1)

    treshLow1 = np.array([0, 0, 30], dtype=np.uint8)
    treshHigh1 = np.array([255, 255, 230], dtype=np.uint8)
    img3 = cv2.inRange(hsvimage, treshLow1, treshHigh1)

    treshLow2 = np.array([0, 0, 0])
    treshHigh2 = np.array([10, 255, 255])
    img4 = cv2.inRange(hsvimage, treshLow2, treshHigh2)

    cv2.bitwise_or(img1,img4,img1)
    cv2.bitwise_and(img1,img2,img1)
    cv2.bitwise_and(img1,img3,img1)

    kernel = np.ones((2,2),np.uint8)
    closing = cv2.morphologyEx(img1,cv2.MORPH_CLOSE,kernel, iterations = 2)
    img1 = cv2.morphologyEx(closing,cv2.MORPH_OPEN,kernel, iterations = 2)


def image_to_feature_vector(image):
    """
    
    :param image: 
    :return: 
    """
def get_hog_features(data_class_id,image_file_name):
    """

    :param data_class_id:The name of directory holding image_feature_filename. (Example = '00000')
    :param image_file_name:The name of image file.(Example = '00000_00001.ppm')
    :return:hog_data_array
    """

    image_feature_filename = image_file_name.replace(".ppm",".txt")
    hog_feature_path = join(hog3_folder,data_class_id,image_feature_filename)
    hog_data_array = np.fromfile(hog_feature_path,dtype=float,count=-1,sep=" ")
    return hog_data_array


if __name__ == "__main__":
    # Test the function read_image_annotations
    data_class_id='00000'
    image_file_name = '00000_00000.ppm'
    hog_feature_data= get_hog_features(data_class_id,image_file_name)
    print(hog_feature_data)
