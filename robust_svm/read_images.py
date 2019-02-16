import csv
import cv2
from os.path import join
import settings
from data_set import *

def make_dataset(data_set_path,data_set_list,hog):
    """
    :param data_set_path: path to datasets(training/test)
    :param data_set_list: folders list
    :return: location
    """
    dataset = '00000'
    annotated_data = read_image_annotations(data_set_path ,data_set_list)
    im = ImageDataset()
    for dataset in data_set_list:
        for each_ann_file in annotated_data:
             if(each_ann_file[file_name] in data_set_path):
                img = cv2.imread(data_set_path+each_ann_file[file_name], cv2.IMREAD_COLOR)
                im.add_image(img, each_ann_file)


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

    datasets_path='../data/training_data/Images/'
    data_set_list = ['00013', '00014', '00015', '00017', '00019']
    make_dataset(datasets_path,data_set_list,False)
