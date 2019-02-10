import csv
import os

def make_dataset(dat_set_path,data_set_list):
    """
    :param dat_set_path: path to datasets(training/test)
    :param data_set_list: folders list
    :return: location
    """
def read_image_annotations(data_path,data_set_name):
    dataSetList = ['00013', '00014', '00015', '00017', '00019']

    dataset = '00000'
    dataloc = 'C:/Users/ch.srivamsi priyanka/Documents/GitHub/traffic_sign_svm/data/training_data/Images/'

    for dataset in dataSetList:

        fileList = os.listdir(dataloc+dataset+'/')

        imageFiles = []
        fileLoc = dataloc+dataset+'/'
        annFile = "GT-"+dataset+".csv"

        annData = []
        with open(fileLoc+annFile, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';')

            count = 0
            for row in spamreader:
                if(count > 0):
                    annData.append(row)
                count = count + 1
        print(annData)






    """
    
    :param data_path: 
    :param data_set_name: 
    :return: 
    """
def get_cropped_image(file_path,annotation):



    """
    
    :param file_path: 
    :param annotation: 
    :return: 
    """
