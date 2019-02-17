import enum as Enum
import numpy as np
from settings import *
from read_images import *
from process_images import get_hog_features


class ImageDataset:

    def __init__(self,data=[],class_ids=[],hog = False):
        self.data=data
        self.class_ids=class_ids
        self.hog=hog

    def add_image(self, data_sets_path, data_set, annotation):
        """
        The function reads file_name and class_id from the annotation file and adds feature_vector using add_row() to data
        :param data_sets_path:Path to our HOG_Features data set.(Used get_hog_features method instead as it holds the path and returns annotations)
        :param data_set:The name of directory holding image_feature_filename. (Example = '00000')
        :param annotation:Annotated Data of single image. Ex:Output of 'read_image_annotations'
        :return:Class instance 'data' is updated
        """

        file_name = annotation[0]
        class_id = annotation[7]
        hog3_feature_vector = get_hog_features(data_set, file_name)
        self.add_row(class_id, hog3_feature_vector)

    def add_row(self, class_id , feature_vector):
        """
        The function reads the image file name and class_id from the annotation data.
        The function reads the corresponding feature file based on the image filename using'get_hog_features'
        and adds it to the instance attribute 'data' using 'add_row'.
        :param class_id:The class_id retrieved from annotations file
        :param feature_vector:The hog_features obtained from add_image method.
        """

        dict = {}
        dict['class_id']= class_id
        dict['feature_vector'] = feature_vector
        self.data.append(dict)

    def shuffle(self):
        """
        The method shuffles instance attribute data using numpy.random.shuffle
        """
        np.random.shuffle(self.data)




class AnnotationMapping(Enum):

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
    im = ImageDataset(data=[2,3],class_ids=[5,6],hog=True)
    im.add_row(class_id=None,feature_vector=None)
    print(im.data)
    print(im.class_ids)
    print(im.hog)
    im.shuffle_indexes()

    # Test ImageDataset.add_image()
    data_sets_path = hog3_folder
    data_set = '00000'
    annotation = read_image_annotations(images_folder, data_set)
    var1 = annotation[-1]
    im.add_image(hog3_folder,data_set,var1)
