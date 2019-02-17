import csv
import cv2
import os
from os.path import join
from settings import *
from data_set import *

def make_dataset(data_set_path,data_set_list,hog):
    """
    The function iterates on the list of datasets and adds the image from the annotation file to the ImageDataset instance via add_image()
    :param data_set_path: path to datasets(training/test)
    :param data_set_list: folders list
    :return: ImageDataset instance
    """

    im = ImageDataset()
    #Iterates on the datasets in the dataset list
    for dataset in data_set_list:
        #Read the annotation file in the dataset
        annotated_data = read_image_annotations(data_set_path ,dataset)
        #Iterates on all annotations in the annotation file
        for each_ann in annotated_data:
            folder_path = join(data_set_path, dataset)
            folder_files = os.listdir(folder_path)
            if(each_ann[0] in folder_files):
                im.add_image(data_set_path,dataset,each_ann)
    return im


def read_image_annotations(datasets_path, dataset_name):
    """
    The function reads the annotation file using csv.reader and returns the annotation data as python list
    :param datasets_path: The path where images and annotation file are present. Example: data/training_data/00001
    :param dataset_name: The directory name. Example:00001
    :return: annotated_data
    """
    annotation_file_folder_path = join(datasets_path, dataset_name)
    annotation_file_name = "GT-{}.csv".format(dataset_name)
    annotation_file_path = join(annotation_file_folder_path , annotation_file_name)
    annotated_data = []
    with open(annotation_file_path, 'r') as csvfile:
        csv_file_reader = csv.reader(csvfile, delimiter=';')
        next(csv_file_reader, None)

        for row in csv_file_reader:
            annotated_data.append(row)

    return annotated_data




def get_cropped_image(file_path,annotation):



    """
    
    :param file_path: 
    :param annotation: 
    :return: 
    """


if __name__ == "__main__":
    # Test the function read_image_annotations

    datasets_path = images_folder
    data_set_list = ['00013','00015']
    make_dataset(datasets_path,data_set_list,False)
