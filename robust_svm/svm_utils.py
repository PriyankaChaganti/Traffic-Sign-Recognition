import os
from read_images import *
from settings import *
from data_set import ImageDataset,AM


def check_image_class(row, image_class):
    """
    The function should check if the row's class_id is the same as the parameter image_class.
    :param row: Row from ImageDataset.(Example:{'class_id': '0', 'feature_vector': ([0.00134514 , 0.706723  ])})
    :param image_class:The class_id of the images which checks if an image belongs to a particular class.
    :return: 1/-1.
    """
    assert type(row['class_id']) == type(image_class)
    if(row['class_id'] == image_class):
        return 1
    else:
        return -1
def test_classifier(multi_class_classifier,test_data):
    """
    
    :param multi_class_classifier: 
    :param test_data: 
    :return: 
    """
if __name__ == "__main__":
   row = {'class_id': '0', 'feature_vector': ([0.00134514, 0.00835472, 0.0826663 , ..., 0.0158771 , 0.137276  ,
       0.706723  ])}
   image_class = '0'
   var2= check_image_class(row,image_class)
   assert var2 ==1
   row = {'class_id': '5', 'feature_vector': ([0.00134514, 0.00835472, 0.0826663 , 0.0158771 , 0.137276])}
   image_class = '2'
   var3= check_image_class(row, image_class)
   assert var3 == -1
