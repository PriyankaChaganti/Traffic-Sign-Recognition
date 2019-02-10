from read_images import read_image_annotations

folderpath="C:/Users/ch.srivamsi priyanka/Documents/GitHub/traffic_sign_svm/data/training_data/Images/"
dataset_name="training_data"
read_image_annotations(folderpath,dataset_name)
"""

read_image_annotations()
training_data=make_dataset(folder_path,folders_list)

test_data=make_dataset(folder_path,folders_list)

multi_class_classifier=multi_class_classifier(training_data,svm_params)

results=test_classifier(multi_class_classifier,test_data)

"""
