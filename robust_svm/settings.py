from os.path import join,isfile,exists
import inspect
hog3_feature_path = "C:/Users/ch.srivamsi priyanka/Documents/GitHub/traffic_sign_svm/data/training_data/Features_HOG/HOG_3"
#hog3_feature_path = os.path.hog3_feature_path(os.path.abspath(inspect.stack()[0][1]))
var1 = isfile(hog3_feature_path)
print(var1)
