import cv2
import getpass
import platform
import datetime
import glob
import numpy as np

#print(getpass.getuser())
#print(platform.uname())
#print(datetime.datetime.now())
#print(cv2.__version__)
path="C:/Users/ch.srivamsi priyanka/Documents/GitHub/traffic_sign_svm/data/training_data/Images/*/*.*"
for file in glob.glob(path):
 a = cv2.imread(file)
 c = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
 cv2.imshow('Color image', c)
 cv2.waitKey(100)
 cv2.destroyAllWindows()

training_data=make_dataset(folder_path,folders_list)

test_data=make_dataset(folder_path,folders_list)

multi_class_classifier=multi_class_classifier(training_data,svm_params)

results=test_classifier(multi_class_classifier,test_data)


