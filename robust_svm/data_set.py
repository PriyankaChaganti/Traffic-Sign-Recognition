from enum import Enum
import numpy as np


class ImageDataset:
    def add_image(self,annotation):
        """

        :param annotation:
        :return:
        """
    def add_row(self,class_id,feature_vector):
        """

        :param class_id:
        :param feature_vector:
        :return:
        """
    def __init__(self):
        self.data=[]
        self.class_ids=[]

    def shuffle_indexes(self):
       # newData = np.random.shuffle(self.data)
        #return newData




class AnnotationMapping(Enum):

    Filename = 0
    Width = 1
    Height = 2
    ROIx1 = 3
    ROIy1 = 4
    ROIx2 = 5
    ROIy2 = 6
    Classid = 7

