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
path="C:/Users/ch.srivamsi priyanka/Documents/GitHub/traffic_sign_svm/data/training_data/*.*"
for file in glob.glob(path):
 a = cv2.imread(file)
 c = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
 cv2.imshow('Color image', c)
 cv2.waitKey(10000)
 cv2.destroyAllWindows()

