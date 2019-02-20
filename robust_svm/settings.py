from os.path import abspath, join, isdir

project_path = abspath(__file__+"/../../")

training_data = join("data", "training_data")
test_data = join("data", "test_data")

images_path = "Images"
hog3_path = join("Features_HOG", "HOG_3")

training_data_folder = join(project_path, training_data)
test_data_folder = join(project_path, training_data)

temp_folder = join(project_path, "temp_space")

if __name__ == "__main__":
    assert isdir(project_path) is True
    assert isdir(training_data_folder) is True
    assert isdir(test_data_folder) is True
    assert isdir(temp_folder) is True
