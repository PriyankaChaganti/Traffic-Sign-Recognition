from robust_svm.read_images import make_dataset
from robust_svm.settings import *

data_path = training_data_folder
feature_path = hog3_path
data_set_list = ['00013', '00015']
training_data = make_dataset(data_path, feature_path, data_set_list)
