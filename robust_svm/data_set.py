from enum import Enum
from random import shuffle
class ImageDataset():
    def add_image(self,data_sets_path,data_set,annotation):
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

        dict = {}
        dict['class_id'] = class_id
        dict['feature_vector'] = feature_vector
        self.data.append(dict)


    def __init__(self,data,class_ids,hog):
        self.data=[]
        self.class_ids=[]
        self.hog=False


    def shuffle_indexes(self):
        newData = []
        ind = range(len(self.data))
        shuffle(ind)
        for i in ind:
            newData.append(self.data[i])
        self.data = newData


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
