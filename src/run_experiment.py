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
        validation_data = bucket
        training_data = combine_data(buckets[:i], buckets[i + 1:])

        # Train the KNN classifier
        # knn_classifier = knn.classify_samples(training_data, training_data.samples, num_neighbors, distance_measure)

        # Train the decision tree classifier
        tree_classifier = DecisionTree(max_depth=max_depth)
        features = find_features_for_continuous_data(training_data)
        featurized_training_dataset = featurize_continuous_data(training_data, features)
        tree_classifier.train(featurized_training_dataset)

        # Get accuracy for both classifiers
        knn_predictions = knn.classify_samples(validation_data.data, validation_data.labels,
                                               num_neighbors, distance_measure)
        tree_predictions = tree_classifier.predict(validation_data.data)

        knn_accuracy = accuracy(knn_predictions, validation_data.labels)
        knn_accuracies.append(knn_accuracy)

        tree_accuracy = accuracy(tree_predictions, validation_data.labels)
        tree_accuracies.append(tree_accuracy)

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
    dt_params = []
    for i in range(1, 101):
        knn_accuracy_euclidean, dt_accuracy = perform_cross_validation(buckets, i, distance_measure=knn.euclidean,
                                                                       max_depth=i)
        knn_params.append((i, knn_accuracy_euclidean, knn.euclidean))
        dt_params.append((i, dt_accuracy))
        knn_accuracy_cosine, dt_accuracy = perform_cross_validation(buckets, i, distance_measure=knn.cosine,
                                                                    max_depth=i)
        knn_params.append((i, knn_accuracy_cosine, knn.cosine))
    knn_params.sort(key=lambda k: k[1])
    dt_params.sort(key=lambda x: x[1])

    best_knn = tuple(knn_params[-1][0][2])
    best_dt = dt_params[-1][0]

    # TODO - evaluate the effectiveness of each classifier on the test set and
    #  generate any other values needed to fill in the assignment 1 report
