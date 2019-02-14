from os.path import join,isfile,exists,abspath
project_path = abspath(__file__+"/../../")
training_data = "data/training_data"
hog3_feature_path = "Features_HOG/HOG_3"
images_path = "Images"
training_data_folder = join(project_path,training_data)
hog3_folder = join(project_path,training_data,hog3_feature_path)
