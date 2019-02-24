from os.path import join, isdir
from os import listdir, mkdir

from robust_svm.data_set import AM
from robust_svm.image_processing_utils import image_to_feature_vector
from robust_svm.read_images import read_image_annotations
from robust_svm.settings import training_data_folder, images_path

data_path = training_data_folder

images_folder = join(data_path, images_path)
features_folder = join(data_path, 'custom_features')

all_files = listdir(images_folder)
#datasets_list = [f for f in all_files if isdir(join(images_folder, f))]
datasets_list = ['00013', '00015']


for dataset in datasets_list:
    print('Creating features for images in dataset {0}'.format(dataset))
    image_dataset_path = join(images_folder, dataset)
    feature_dataset_path = join(features_folder, dataset)
    if not isdir(feature_dataset_path):
        mkdir(feature_dataset_path)

    annotated_data = read_image_annotations(image_dataset_path, dataset)
    all_images = listdir(image_dataset_path)
    for each_annotation in annotated_data:
        image_filename = each_annotation[AM.Filename]
        if image_filename in all_images:
            feature_vector = image_to_feature_vector(image_dataset_path, each_annotation)
            feature_file_name = image_filename.replace('.ppm', '.txt')
            feature_file_path = join(feature_dataset_path, feature_file_name)
            with open(feature_file_path, "w") as f:
                for each_val in feature_vector:
                    f.write(str(each_val) + "\n")
