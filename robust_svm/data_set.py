import numpy as np

from robust_svm.process_images import read_feature_file
from robust_svm.settings import *


class ImageDataset:
    def __init__(self, images_path=None, features_path=None):
        self.data = []
        self.class_ids = []
        self.images_path = images_path
        self.features_path = features_path

    def add_image(self, data_set, annotation):
        """
        The function reads file_name and class_id from the annotation file and adds feature_vector using add_row() to data
        :param data_sets_path:Path to our HOG_Features data set.(Used get_hog_features method instead as it holds the path and returns annotations)
        :param data_set:The name of directory holding image_feature_filename. (Example = '00000')
        :param annotation:Annotated Data of single image. Ex:Output of 'read_image_annotations'
        """

        file_name = annotation[AM.Filename]
        class_id = annotation[AM.Classid]
        feature_vector = read_feature_file(self.features_path, data_set, file_name)
        self.add_row(class_id, feature_vector)

    def add_row(self, class_id, feature_vector):
        """
        The function creates a dictionary with keys 'class_id and 'feature_vector' and appends the dictionary to instance attribute data.
        :param class_id:The class_id retrieved from annotations file
        :param feature_vector:The hog_features obtained from add_image method.
        """
        row = dict()
        row['class_id'] = class_id
        row['feature_vector'] = feature_vector
        self.data.append(row)

    def shuffle(self):
        """
        The method shuffles instance attribute data using numpy.random.shuffle
        """
        np.random.shuffle(self.data)


class AM:
    Filename = 0
    Width = 1
    Height = 2
    ROIx1 = 3
    ROIy1 = 4
    ROIx2 = 5
    ROIy2 = 6
    Classid = 7


if __name__ == '__main__':
    # Test ImageDataset instantiation
    images_folder_path = join(training_data_folder, images_path)
    features_folder_path = join(test_data_folder, hog3_path)
    im = ImageDataset(images_path=images_folder_path, features_path=features_folder_path)

    # Test ImageDataset.add_image()
    sample_data_set = '00000'
    from robust_svm.read_images import read_image_annotations
    dataset_folder_path = join(images_folder_path, sample_data_set)
    image_annotations = read_image_annotations(dataset_folder_path, sample_data_set)
    sample_annotation = image_annotations[-1]
    im.add_image(sample_data_set, sample_annotation)

    # Test ImageDataset.add_row()
    sample_class_id = '1'
    sample_feature_vector = [1, 2, 3]
    im.add_row(sample_class_id, sample_feature_vector)
