from enum import Enum
import numpy as np
class ImageDataset():
    def add_image(self,data_sets_path,data_set,annotation):
        """

        :param data_sets_path:
        :param data_set:
        :param annotation:
        :return:
        """


    def add_row(self,class_id,feature_vector):
        """

        :param class_id:
        :param feature_vector:
        :return:
        """

        dict = {}
        dict['class_id'] = class_id
        dict['feature_vector'] = feature_vector
        self.data.append(dict)


    def __init__(self,data,class_ids,hog = False):
        self.data=[]
        self.class_ids=[]


    def shuffle_indexes(self):
        newData = np.random.shuffle(self.data)
        return newData




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
    im = ImageDataset(data=[100,200], class_ids=[1,2], hog=False)
    print(im.data)
    print(im.class_ids)
    print(im.hog)
