import os
from read_images import *
from settings import *
from data_set import ImageDataset,AM


def check_image_class(row, image_class):
    """
    
    :param image_data: 
    :param expected_image_class: 
    :return: 
    """
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
   im = ImageDataset(data=[2,3],class_ids=[5,6],hog=True)
   im.add_row(class_id=None,feature_vector=None)
   data_sets_path = hog3_folder
   data_set = '00000'
   var1 = ['00006_00029.ppm', '112', '118', '10', '11', '103', '108', '0']
   im.add_image(hog3_folder,data_set,var1)
   row = {'class_id': '0', 'feature_vector': ([0.00134514, 0.00835472, 0.0826663 , ..., 0.0158771 , 0.137276  ,
       0.706723  ])}
   image_class = '0'
   var2= check_image_class(row,image_class)
   print(var2)
