from read_images import make_dataset
from settings import *

#folderpath="C:/Users/ch.srivamsi priyanka/Documents/GitHub/traffic_sign_svm/data/training_data/Images/"
#dataset_name="training_data"
#read_image_annotations(folderpath,dataset_name)

data_set_list = ['00013','00015']
data_set_path = images_folder
hog = None
training_data = make_dataset(data_set_path,data_set_list,hog)
"""

read_image_annotations()
training_data=make_dataset(folder_path,folders_list)

test_data=make_dataset(folder_path,folders_list)

multi_class_classifier=multi_class_classifier(training_data,svm_params)

results=test_classifier(multi_class_classifier,test_data)

"""
