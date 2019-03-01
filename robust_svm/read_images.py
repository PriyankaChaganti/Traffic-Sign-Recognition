import csv
from functools import wraps
from os import listdir
from time import time


from robust_svm.data_set import *


def make_dataset(data_path, feature_path, data_set_list):
    """
    The function iterates on the list of datasets and adds the image from the annotation
    file to the ImageDataset instance via add_image()
    :param data_path: path to datasets(training/test)
    :param feature_path: path to features folder relative to data_path
    :param data_set_list: folders list
    :return: ImageDataset instance
    """
    images_folder_path = join(data_path, images_path)
    features_folder_path = join(data_path, feature_path)
    im = ImageDataset(images_path=images_folder_path, features_path=features_folder_path)

    #Iterates on the datasets in the dataset list
    for dataset in data_set_list:
        start_time = time()
        folder_path = join(images_folder_path, dataset)
        #Read the annotation file in the dataset
        annotated_data = read_image_annotations(folder_path, dataset)
        #Iterates on all annotations in the annotation file
        for eachann in annotated_data:
            folder_files = listdir(folder_path)
            if(eachann[AM.Filename] in folder_files):
                im.add_image(dataset, eachann)
        time_diff = time() - start_time
        print("Loaded image features at {0} for dataset {1}. Time taken: {2} seconds".format(
            feature_path, dataset, time_diff))
    return im


def read_image_annotations(folder_path, dataset_name):
    """
    The function reads the annotation file using csv.reader and returns the annotation data as python list
    :param folder_path: The path where images and annotation file are present. Example: data/training_data/00001
    :param dataset_name: The directory name. Example:00001
    :return: annotated_data
    """
    annotation_file_name = "GT-{}.csv".format(dataset_name)
    annotation_file_path = join(folder_path, annotation_file_name)
    annotated_data = []
    with open(annotation_file_path, 'r') as csvfile:
        csv_file_reader = csv.reader(csvfile, delimiter=';')
        next(csv_file_reader, None)
        for row in csv_file_reader:
            annotated_data.append(row)
    return annotated_data


def get_all_datasets(data_path):
    """
    Returns a list of all datasets in a data path
    :param data_path: Path to training or test data (Ex:traffic_sign_svm\data\training_data)
    :return: list of datasets (Ex: [00013, 00015])
    """
    images_folder = join(data_path, images_path)
    all_files = listdir(images_folder)
    all_datasets = [d for d in all_files if isdir(d)]
    return all_datasets


if __name__ == "__main__":
    # Test the function read_image_annotations
    sample_feature_path = join(training_data_folder, hog3_path)
    sample_data_set_list = ['00013', '00015']
    test_datset = make_dataset(training_data_folder, sample_feature_path, sample_data_set_list)
