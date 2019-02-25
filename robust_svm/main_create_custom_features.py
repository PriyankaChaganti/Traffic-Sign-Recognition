from os.path import join, isdir
from os import listdir, mkdir

from robust_svm.data_set import AM
from robust_svm.image_processing_utils import image_to_feature_vector
from robust_svm.read_images import read_image_annotations, get_all_datasets
from robust_svm.settings import training_data_folder, test_data_folder, images_path


data_paths = [training_data_folder, test_data_folder]

for data_path in data_paths:
    images_folder = join(data_path, images_path)
    features_folder = join(data_path, 'custom_features')
    # datasets_list = get_all_datasets(data_path)
    datasets_list = ['00013', '00015']
    for dataset in datasets_list:
        image_dataset_path = join(images_folder, dataset)
        feature_dataset_path = join(features_folder, dataset)
        print('Creating features at {0} for images in {1}'.format(
            feature_dataset_path, image_dataset_path))

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
