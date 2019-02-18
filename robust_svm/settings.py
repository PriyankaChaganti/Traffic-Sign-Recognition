from os.path import join,abspath

project_path = abspath(__file__+"/../../")
training_data = join("data", "training_data")
hog3_path = join("Features_HOG", "HOG_3")
images_path = "Images"
training_data_folder = join(project_path,training_data)
hog3_folder = join(training_data_folder,hog3_path)
images_folder = join(training_data_folder,images_path)
temp_folder = join(project_path,"temp_space")
print(images_folder)
