import pickle
from robust_svm.read_images import make_dataset
from robust_svm.settings import *

data_path = training_data_folder
feature_path = hog3_path
data_set_list = ['00013', '00015']
training_data = make_dataset(data_path, feature_path, data_set_list)

training_data = pickle.dump(training_data, open( "../data/dumps/training_data.p", "wb" ))
training_data_unpickle = pickle.load( open( "../data/dumps/training_data.p", "rb" ) )

if(training_data == training_data_unpickle):
    print("Same")
else:
    print("Not Same")
