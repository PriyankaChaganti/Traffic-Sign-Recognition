from robust_svm.read_images import *
from robust_svm import settings
import pickle


def determine_image_class(multiple_labels):
    labels_sum = sum(multiple_labels.values())

    if labels_sum == 1:
        for class_id, label in multiple_labels.items():
            if label == 1:
                return class_id

    return None


def test_classifier(multi_class_classifier, test_data):
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
