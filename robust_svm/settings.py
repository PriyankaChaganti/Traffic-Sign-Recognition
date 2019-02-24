from os.path import abspath, join, isdir

project_path = abspath(__file__+"/../../")

training_data_path = join("data", "training_data")
test_data_path = join("data", "test_data")
dumps_path = join("data", "dumps")

images_path = "Images"
hog1_path = join("Features_HOG", "HOG_1")
hog2_path = join("Features_HOG", "HOG_2")
hog3_path = join("Features_HOG", "HOG_3")
custom_features_path = "custom_features"

training_data_folder = join(project_path, training_data_path)
test_data_folder = join(project_path, test_data_path)
dumps_folder = join(project_path, dumps_path)

temp_folder = join(project_path, "temp_space")

if __name__ == "__main__":
    assert isdir(project_path) is True
    assert isdir(training_data_folder) is True
    assert isdir(test_data_folder) is True
    assert isdir(dumps_folder) is True
    assert isdir(temp_folder) is True
    assert isdir(join(training_data_folder, images_path)) is True
    assert isdir(join(test_data_folder, images_path)) is True
    assert isdir(join(training_data_folder, hog3_path))is True
    assert isdir(join(test_data_folder, hog3_path)) is True
