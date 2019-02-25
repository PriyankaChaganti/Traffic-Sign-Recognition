from robust_svm.read_images import *
from robust_svm import settings
import pickle


def check_image_class(row, image_class):
    """
    The function should check if the row's class_id is the same as the parameter image_class.
    :param row: Row from ImageDataset.(Example:{'class_id': '0', 'feature_vector': ([0.00134514 , 0.706723  ])})
    :param image_class:The class_id of the images which checks if an image belongs to a particular class.
    :return: 1/-1.
    """
    assert type(row['class_id']) == type(image_class)
    if row['class_id'] == image_class:
        return 1
    else:
        return -1


def determine_image_class(multiple_labels):
    """
    The function determines if the given image is classified to a particular class.
    :param multiple_labels:Output of get_all_svm_labels.
    :return:class_id/None
    """
    labels_sum = sum(multiple_labels.values())

    if labels_sum == 1:
        for class_id, label in multiple_labels.items():
            if label == 1:
                return class_id
    return None


def test_classifier(multi_class_classifier, test_data):
    """
    The function classifies test_data with the help of the classifier and also checks how accurately the images are classified.
    :param: multi_class_classifier:The classifier which is trained with data.
    :param: test_data:The data which has to be classified.
    :return:accuracy_results
    """
    print('Testing the classifier')
    accuracy_results = {}
    for class_id in test_data.class_ids:
        accuracy_results[class_id] = {'right': 0, 'wrong': 0, 'total': 0}

    for eachRow in test_data.data:
        results = multi_class_classifier.get_all_classifier_labels(eachRow)
        classifier_output_class_id = determine_image_class(results)
        expected_class_id = eachRow['class_id']

        if expected_class_id == classifier_output_class_id:
            accuracy_results[expected_class_id]['right'] += 1
        else:
            accuracy_results[expected_class_id]['wrong'] += 1

        accuracy_results[expected_class_id]['total'] += 1
    return accuracy_results


if __name__ == "__main__":
    # Test the method check_image_class
    sample_row = {'class_id': '0', 'feature_vector': ([0.00134514, 0.00835472, 0.0826663, 0.0158771, 0.137276])}
    sample_image_class = '0'
    var2 = check_image_class(sample_row, sample_image_class)
    assert var2 == 1
    sample_row_2 = {'class_id': '5', 'feature_vector': ([0.00134514, 0.00835472, 0.0826663, 0.0158771, 0.137276])}
    image_class_2 = '2'
    var3 = check_image_class(sample_row_2, image_class_2)
    assert var3 == -1

    # Test the method test_classifier
    data_set_list = ['00013', '00015']
    training_data_pickle_path = join(settings.dumps_folder, "training_data.p")
    training_data = pickle.load(open(training_data_pickle_path, "rb"))

    multi_svm_classifier_pickle_path = join(settings.dumps_folder, 'multi_svm_classifier.p')
    multi_svm_classifier = pickle.load(open(multi_svm_classifier_pickle_path, "rb"))

    test_data_pickle_path = join(settings.dumps_folder, "test_data.p")
    test_data = pickle.load(open(test_data_pickle_path, "rb"))

    label_sum = test_classifier(training_data,test_data)
    print(label_sum)
