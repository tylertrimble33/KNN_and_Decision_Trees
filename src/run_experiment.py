from src.decision_tree import *
import src.nearest_neighbor as knn
import numpy as np


def accuracy(predicted_labels, true_labels):
    """
    Calculates accuracy of predicted labels
    :param predicted_labels: a numpy array of predicted labels
    :param true_labels: a numpy array of true labels
    :return: the accuracy of the predicted labels
    """
    # TODO - implement me
    return np.mean(predicted_labels == true_labels)


def perform_cross_validation(buckets, num_neighbors=1, distance_measure=knn.euclidean, max_depth=9999999):
    """
    Performs cross validation on a knn classifier and a decision tree classifier using
    the specified hyper-parameters

    :param buckets: the divided training dataset
    :param num_neighbors: the number of neighbors in knn
    :param distance_measure: the distance mesure for knn
    :param max_depth: the maximum depth for decision tree
    :return: the average validation accuracy over all folds for the
             knn classifier and decision tree respectively
    """
    # TODO - implement me
    knn_accuracies = []
    tree_accuracies = []
    for i, bucket in enumerate(buckets):
        training_data, validation_data = get_fold_train_and_validation(buckets, i)

        # Train the KNN classifier
        knn_classifier = knn.classify_samples(training_data, training_data.samples, num_neighbors, distance_measure)

        # Train the decision tree classifier
        tree_classifier = DecisionTree(max_depth=max_depth)
        features = find_features_for_continuous_data(training_data)
        featurized_training_dataset = featurize_continuous_data(training_data, features)
        # tree_classifier.train(featurized_training_dataset)

        # Get accuracy for both classifiers
        knn_predictions = knn.classify_samples(validation_data, validation_data.samples,
                                               num_neighbors, distance_measure)
        # tree_predictions = tree_classifier.predict(validation_data.data)

        knn_accuracy = accuracy(knn_predictions, validation_data.labels)
        knn_accuracies.append(knn_accuracy)

        # tree_accuracy = accuracy(tree_predictions, validation_data.labels)
        # tree_accuracies.append(tree_accuracy)

    return np.mean(knn_accuracies), np.mean(tree_accuracies)


if __name__ == '__main__':
    # hard coded parameters
    num_folds = 5

    # load breast cancer data from file
    raw_data = Dataset.load_from_file(os.path.join("..", "data", "breast-cancer-wisconsin.csv"))

    # shuffle data before split
    raw_data.shuffle()

    # Perform a test/train split
    test_data, training_data = raw_data.test_train_split(20)

    # create each bucket
    buckets = training_data.split_into_folds(num_folds)

    # TODO - implement methods in marked by TODOs in dataset.py
    #  These will be used for cross-validation

    # TODO - implement a hyper-parameter search using average accuracy over
    #  all folds of cross validation as the evaluation metric.
    #  For KNN, check k=1 to 100 with euclidean and cosine distance
    #  For decision trees, check max_depth = 1 to 100
    knn_params = []
    # dt_params = []
    for i in range(1, 101):
        knn_accuracy_euclidean, dt_accuracy = perform_cross_validation(buckets, i, distance_measure=knn.euclidean,
                                                                       max_depth=i)
        knn_params.append((i, knn_accuracy_euclidean, "Euclidean"))
        # dt_params.append((i, dt_accuracy))
        knn_accuracy_cosine, dt_accuracy = perform_cross_validation(buckets, i, distance_measure=knn.cosine,
                                                                    max_depth=i)
        knn_params.append((i, knn_accuracy_cosine, "Cosine"))
        print("Iteration Complete: " + str(i))
    knn_params.sort(key=lambda k: k[1])
    # dt_params.sort(key=lambda x: x[1])

    # print(knn_params)
    # best_dt = dt_params[-1][0]

    knn_train_samples = knn.classify_samples(training_data, training_data.samples, 2, knn.cosine)
    print("KNN Training accuracy: " + str(accuracy(knn_train_samples, training_data.labels)))
    knn_test_samples = knn.classify_samples(test_data, test_data.samples, 2, knn.cosine)
    print("KNN Test accuracy: " + str(accuracy(knn_test_samples, test_data.labels)))

    # TODO - evaluate the effectiveness of each classifier on the test set and
    #  generate any other values needed to fill in the assignment 1 report
    def find_feature_with_highest_covariance(dataset):
        covariance_matrix = np.cov(dataset.samples, rowvar=False)
        max_covariance = 0
        max_feature = 0
        for i in range(covariance_matrix.shape[0]):
            covariance = np.max(covariance_matrix[i, :])
            if covariance > max_covariance:
                max_covariance = covariance
                max_feature = i
        return max_feature
    print("Covariance: " + str(find_feature_with_highest_covariance(raw_data)))


    def find_feature_with_highest_variance(dataset):
        variances = np.var(dataset.samples, axis=0)
        max_variance = np.max(variances)
        max_feature = np.argmax(variances)
        return max_feature
    print("Variance: " + str(find_feature_with_highest_variance(raw_data)))

    def majority_class(dataset):
        zeros = 0
        ones = 0
        for i in dataset.labels:
            if i == 0:
                zeros += 1
            else:
                ones += 1
        print("Zero: " + str(zeros))
        print("Ones: " + str(ones))
    majority_class(raw_data)
