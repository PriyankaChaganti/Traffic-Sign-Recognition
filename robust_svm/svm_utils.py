from robust_svm.read_images import *
from robust_svm.settings import *
import pickle



def check_image_class(row, image_class):
    """
    The function should check if the row's class_id is the same as the parameter image_class.
    :param row: Row from ImageDataset.(Example:{'class_id': '0', 'feature_vector': ([0.00134514 , 0.706723  ])})
    :param image_class:The class_id of the images which checks if an image belongs to a particular class.
    :return: 1/-1.
    """
    assert type(row['class_id']) == type(image_class)
    if(row['class_id'] == image_class):
        return 1
    else:
        return -1
def test_classifier(multi_class_classifier,test_data):
    """
    
    :param multi_class_classifier: 
    :param test_data: 
    :return: 
    """
    svm_label = {}
    for eachrow in test_data:
        svm_label = ml.get_all_classifier_labels(eachrow)
        svm_label_values = svm_label.values()
        label_sum = sum(svm_label_values)
    return label_sum





if __name__ == "__main__":
   from robust_svm.classifier import MultiClassClassifier

   row = {'class_id': '0', 'feature_vector': ([0.00134514, 0.00835472, 0.0826663 , ..., 0.0158771 , 0.137276  ,
       0.706723  ])}
   image_class = '0'
   var2= check_image_class(row,image_class)
   assert var2 ==1
   row = {'class_id': '5', 'feature_vector': ([0.00134514, 0.00835472, 0.0826663 , 0.0158771 , 0.137276])}
   image_class = '2'
   var3= check_image_class(row, image_class)
   assert var3 == -1

   row = {'0013': 1, '0015': 0, '0017':0}
   data_path = training_data_folder
   feature_path = hog3_path
   data_set_list = ['00013', '00015']
   training_data = pickle.load( open( "../data/dumps/training_data.p", "rb" ) )
   kernel_type = "linear"
   ml = MultiClassClassifier(training_data,1,{"r0":1,"c":1,kernel_type:"linear"})
   test_data = join(test_data_folder,hog3_path)
   label_sum = test_classifier(training_data,test_data)
   print(label_sum)


