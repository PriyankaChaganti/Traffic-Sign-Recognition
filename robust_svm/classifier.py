import numpy
from robust_svm.svm_utils import check_image_class
from robust_svm.data_set import *


class MultiClassClassifier:
    """SVM-based multi class classifier"""
    def __init__(self, training_data, epochs, svm_params):
        """
        :param training_data: Data for training the svm classifier. Instance of ImageDataset class.
        :param epochs: Number of times the training data should be fed to the classifier.
        :param svm_params: SVM parameters such as r0: rate of convergence, C: constant,
        kernel_type: type of the kernel used in the svm
        """
        self.classifiers = {}
        self.epochs = epochs  # Number of times data must be fed to the svm
        self.r0 = svm_params.get('r0', 0.6)  # Rate of convergence
        self.C = svm_params.get('C', 0.9)  # Constant
        self.kernel_type = svm_params.get('kernel_type', 'linear')

        # Sigmoid kernel parameters. These parameters are required only if
        # self.kernel_type is 'sigmoid'
        self.s1 = svm_params.get('s1', 0.0001)
        self.s2 = svm_params.get('s2', 0.85)

        # Polynomial kernel parameters. These parameters are required only
        # if self.kernel_type is 'polynomial'
        self.p1 = svm_params.get('p1', 1)
        self.p2 = svm_params.get('p2', 2)
        self.p3 = svm_params.get('p3', 3)

        # Build the classifier using training_data
        print('Building the classifier using training data')
        self.build_classifier(training_data)

    def kernel(self, w, row):
        """
        The function performs various operations based on the kernel selected.
        :param w: weights vector
        :param row: individual data point in the data
        :return: dot-product of weight vector and feature vector
        """
        # Returns (W.R)
        if self.kernel_type == "linear":
            y = numpy.dot(numpy.transpose(w), row['feature_vector'])
            return y

        # Returns tan((c1 * (W.R)) - c2)
        if self.kernel_type == "sigmoid":
            y = numpy.tanh(self.s1*numpy.dot(numpy.transpose(w), row['feature_vector']) - self.s2)
            return y

        # Returns (c1 *(W.R) + c2)^c3
        if self.kernel_type == "polynomial":
            y = (1*numpy.dot(numpy.transpose(w), row['feature_vector']) + 2)**3
            return y

    def build_classifier(self, training_data):
        """Builds a multi-class classifier by training the classifier with
        each type of dataset individually. The function adds the classifiers to the
        self.classifiers as a dictionary where the keys of the dictionary are the class_id s
        of the data. Example: If the training data has two datasets with class ids 9, 13.
        self.classifiers will have {9: <weight_vector_1>, 13:<weight_vector_2>}
        <weight_vector_1> separates the images with class id 9 from the rest of the images
        with a different class id
        <weight_vector_2> separates the images with class id 13 from the rest of the images
        with a different class id
        :param training_data: training data to build the multi class classifier.
        An instance of ImageDatset class
        :return: Doesn't return anything. Just updates self.classifiers
        """
        for class_id in training_data.class_ids:
            w = self.train_classifier(training_data, class_id)
            self.classifiers[class_id] = w

    def train_classifier(self, training_data, training_class_id):
        """
        This function takes training_data and the class_id of the data to train on.
        All the images with their class id same as the param class_id are considered as
        "positive" data and all the images with any other class id are considered as
        "negative data" for training
        :param training_data: Data for training. An instance of ImageDatset class
        :param training_class_id: a class id of the images that will be as "positive" data for
        the current classifier
        :return: Returns a weight vector (i.e classifier) that identifies the images with
        class id, training_class_id
        """
        w = numpy.array([0] * len(training_data.data[0]['feature_vector']))
        t = 0
        for eachEpoch in range(self.epochs):
            training_data.shuffle()
            for row in training_data.data:
                label = check_image_class(row, training_class_id)
                rate = self.r0/(1 + ((self.r0*t)/self.C))

                if label*self.kernel(w, row) <= 1:
                    try:
                        delJ = w - (self.C*(label*row['feature_vector']))
                    except ValueError as ex:
                        pass
                else:
                    delJ = w

                w = w - (rate*delJ)
                t = t + 1
        return w

    def get_svm_label(self, classifier_id, row):
        """
        The function calculates the kernel value using the classifier(w) and feature_vector(row) and returns a
        label which could be 1 or 0. The function returns 1 if the kernel value is >0. It returns 0 if the
        kernel value is <=0. The kernel returns +ve value if the feature_vector belongs to the
        classification represented by the classifier. The kernel returns -ve value if the feature_vector does
        not belong to the classification represented by the classifier.
        :param classifier_id: Id or key of the classifier in self.classifiers
        :param row: data point or feature vector for which the label must be determined
        :return: label
        """
        w = self.classifiers[classifier_id]
        y = self.kernel(w, row)
        if y > 0:
            label = 1
        else:
            label = 0

        return label

    def get_all_classifier_labels(self,row):
        """
        The function gets the label of the row for every classifier and returns the label as a dictionary.
        :param row:The data point which has to be labelled.
        :return: results
        """

        results = {}
        for eachclassifier in self.classifiers:
            svm_label = self.get_svm_label(eachclassifier, row)
            results[eachclassifier] = svm_label
        return results


if __name__ == "__main__":
    import pickle
    import settings
    training_data_pickle_path = join(settings.dumps_folder, "training_data.p")
    multi_svm_classifier_pickle_path = join(settings.dumps_folder, 'multi_svm_classifier.p')
    feature_path = hog3_path
    data_set_list = ['00013', '00015']
    training_data = pickle.load(open(training_data_pickle_path, "rb"))
    svmclassifier = pickle.load(open(multi_svm_classifier_pickle_path, "rb"))
    #svmclassifier = MultiClassClassifier(training_data,1,{'r0':1,'C':1,"kernel_type":'linear'})
    #ml.build_classifier(training_data)
    row = training_data.data[0]
    results = svmclassifier.get_all_classifier_labels(row)
    print(results)
