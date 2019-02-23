import pickle
from robust_svm.read_images import make_dataset
from robust_svm.settings import *
from robust_svm.classifier import MultiClassClassifier

data_path = training_data_folder
feature_path = hog3_path
data_set_list = ['trial_dataset']
training_data = make_dataset(data_path, feature_path, data_set_list)
epoch = 1
kernel_type = 'linear'
svmclassifier = MultiClassClassifier(training_data,epoch,{"r0":1,"c":1,kernel_type:"linear"})

svmclassifier_pickle = pickle.dump(svmclassifier,open("../data/dumps/svmclassifier.p","wb"))
svmclassifier_unpickle = pickle.load(open("../data/dumps/svmclassifier.p","rb"))
test_data = make_dataset(test_data_folder,hog3_path,data_set_list)

training_data_pickle = pickle.dump(training_data, open( "../data/dumps/training_data.p", "wb" ))
training_data_unpickle = pickle.load( open( "../data/dumps/training_data.p", "rb" ) )



