import numpy as np
from src.dataset import *
import matplotlib.pyplot as plt
import os
import random


def find_features_for_continuous_data(data):
    """
    Finds features for continuous data - features are halfway points
    between class boundaries

    :param data: the data to create features for
    :return: a list of lists describing the split values for each dimension
             the first dimension contains a list for each dimension of the original data
             each of those lists contains a single list of values which describe where
             to split the data in that dimension. For example, a return value of
             [[1.2,3.7],[2.9]] indicates that 3 features were found. The first feature
             feature asks the question "Is dimension 1 less than 1.2?". The second feature
             asks the question "Is dimension 1 less than 3.5?", and the third feature asks
             the question "Is dimension 2 less than 2.9?"
    """

    # TODO - implement me
    # Hint: In my implementation I created a dictionary of value-label pairs for each feature
    #       which I called points. I sorted the points by their value:
    #            sorted_points = sorted(points.items(), key=lambda item: item[0]))
    #       Then, if there labels changed, I found a feature as the midpoint between the two
    #       values.

    # Minor Implementation note: When creating the dictionary of points there may be collisions
    # where is already in the dictionary (possibly with a different label). It is OK to ignore
    # these collisions (just overwrite the previous value).

    features = []

    for i in range(data.num_features):
        points = {}
        for j in range(data.num_samples):
            points[data.samples[j][i]] = data.labels[j]

        sorted_points = sorted(points.items(), key=lambda item: item[0])
        split_values = []

        for j in range(1, len(sorted_points)):
            if sorted_points[j][1] != sorted_points[j - 1][1]:
                split_values.append((sorted_points[j][0] + sorted_points[j - 1][0]) / 2)
        features.append(split_values)
    return features


def featurize_continuous_data(data, features):
    """
    Featurizes a dataset based on the features described in features

    :param data: a dataset object
    :param features: features generated from find_features_for_continuous_data
    :return: the featurized dataset
    """
    # TODO implement me
    # Hint: Recall, you create a set of features which indicates the feature number
    #       and value for a "is sample[feature_number] < value" type test. This method
    #       converts numeric data to binary (True/False) data using those features

    featurized_samples = np.zeros((data.num_samples, len(features)))

    for i in range(data.num_samples):
        for j, split_values in enumerate(features):
            for split_value in split_values:
                if data.samples[i][j] < split_value:
                    featurized_samples[i][j] = 1
                    break

    return Dataset(featurized_samples, data.labels)


def entropy(dataset):
    """
    Calculates entropy of a dataset

    :param dataset: a dataset object
    :return: the entropy of the dataset
    """
    # TODO - implement me
    counts = [0] * len(dataset.labels)
    total = len(dataset.samples)

    for sample in range(len(dataset.samples)):
        counts[int(dataset.labels[sample])] += 1
    ent = 0

    for count in counts:
        prob = count / total
        if prob > 0:
            ent -= prob * np.log2(prob)
    return ent


class Node:
    def __init__(self, data=None):
        """
        Creates a new node object.

        Each node contains a dataset, and optionally a test and a true and false child node.
        The dataset is stored in the data parameter.
        Recall that data is featurized as True/False features, so the test stored at this node
        checks (if samples[feature_index] == True). The dataset is split on this test, and
        samples which are true for that feature are stored in true_child, and features which
        are false are stored in false child.

        :param data: the data this node contains
        """
        self.data = data
        self.feature_index = None
        self.true_child = None
        self.false_child = None

    def __str__(self):
        """
        converts this node to a string
        :return: a string representation of the node
        """
        return self.data.class_ratio_string() + " split by feature " + str(self.feature_index)

    def split_by(self, feature_index):
        """
        Splits the data contained in this node by the binary feature at feature_index
        :param feature_index: the feature_index to split on
        :return: a tuple consisting of (data_true, data_false) - where data_true contains
                 samples that are True at feature_index and data_false contains samples
                 that are false at data_index
        """
        # TODO - implement me
        #  Hint: This is a helper function for find_feature_which_best_spits.
        #  It splits the dataset at this node into samples which are
        #  true vs. false for the feature at feature_index
        data_true = []
        data_true_labels = []
        data_false = []
        data_false_labels = []
        data = self.data.samples

        for sample in range(len(data)):
            if data[sample][feature_index]:
                data_true.append(data[sample])
                data_true_labels.append(self.data.labels[sample])
            else:
                data_false.append(data[sample])
                data_false_labels.append(self.data.labels[sample])
        return data_true, data_true_labels, data_false, data_false_labels

    def find_feature_which_best_splits(self):
        """
        Finds the feature which best splits the data in this node using
        information gain as the splitting criteria

        :return: the index of the feature that best splits the data
                 returns -1 if no feature increases information
        """
        # TODO - implement me
        best_feature = -1
        best_information_gain = 0
        current_entropy = entropy(self.data)

        for feature_idx in range(self.data.num_features):
            true_data, true_data_labels, false_data, false_data_labels = self.split_by(feature_idx)
            true_data_probability = len(true_data) / len(self.data.samples)
            true_data_entropy = entropy(Dataset(true_data, true_data_labels))

            false_data_probability = len(false_data) / len(self.data.samples)
            false_data_entropy = entropy(Dataset(false_data, false_data_labels))

            information_gain = current_entropy - (true_data_probability * true_data_entropy) - (
                    false_data_probability * false_data_entropy)

            if information_gain > best_information_gain:
                best_feature = feature_idx
                best_information_gain = information_gain

        return best_feature


class DecisionTree:
    def __init__(self, max_depth=100):
        """
        Constructs a new DecisionTree object
        :param max_depth: the maximum depth the DecisionTree will reach during training
        """
        self.root = Node()
        self.max_depth = max_depth
        self.features = None

    def predict(self, data):
        """
        Uses the trained decision tree to predict the class of each
        sample in data
        :param data: the data to classify
        :return: a numpy array of binary labels corresponding to the
                 prediction of each sample in data
        """
        # TODO - implement me
        predictions = []

        for sample in data.samples:
            predictions.append(self._predict_sample(sample, self.root))
        return np.array(predictions)

    def _predict_sample(self, sample, current_node):
        """
        Recursive method for predicting the class of a sample.

        :param sample: the sample to predict
        :param current_node: the current node in the decision tree
        :return: True/False indicating the class label of the sample
        """
        # TODO implement me
        #  Hint: to perform prediction, traverse the tree with the sample.
        #  The final prediction should be the majority class of leaf node
        #  reached by traversing the tree with the sample.
        # if sample < current_node.feature_index:
        #     print("Sample: " + str(sample))
        if current_node.true_child is None:
            return current_node.data.get_majority_class()
        elif sample[current_node.feature_index]:
            return self._predict_sample(sample, current_node.false_child)
        else:
            return self._predict_sample(sample, current_node.true_child)
        # print(featurize_continuous_data(current_node.data, self.features).labels.max())
        # print(current_node.)

    def __str__(self):
        """
        Creates a string representation of this DecisionTree
        :return:
        """
        return self._recursive_string(self.root, "")

    def _recursive_string(self, node, leading_tabs):
        """
        Recursive helper function for __str__
        :param leading_tabs: the number of tabs to indent
        :return: a string representation of this DecisionTree
        """
        string = leading_tabs + str(node)
        if node.true_child is not None:
            string += "\n"
            string += self._recursive_string(node.true_child, leading_tabs + "\t")
            string += "\n"
            string += self._recursive_string(node.false_child, leading_tabs + "\t")
        return string

    def train(self, data):
        """
        Creates a decision tree for classifying the data
        :param data: the data to classify
        :return: None
        """
        self.root.data = data
        self._divide(self.root, 0)

        print("Training Complete")
        print(str(self))

    def _divide(self, current_node, depth):
        """
        Recursive helper method for Train - constructs the decision tree by
        recursively dividing the training data and storing the split decisions
        at nodes in the tree. Remember, datasets are divided based on a sample's
        feature value, NOT the class value

        :param current_node: The current node to be split
        :param depth: the depth of the current node
        :return: None - the constructed decision tree is stored starting with self.root
        """
        # TODO - implement me
        #  Hint: this is the recursive method that is used to train the decision tree.
        #   Remember the base cases, which are:
        #      all classes are the same
        #      splitting by a feature adds no information
        #      depth > max depth

        if depth > self.max_depth:
            return
        if current_node.data.all_classes_same():
            return
        if current_node.find_feature_which_best_splits() == -1:
            return
        else:

            split_feature = current_node.find_feature_which_best_splits()
            current_node.feature_index = split_feature
            true_data, true_data_labels, false_data, false_data_labels = current_node.split_by(current_node.feature_index)

            current_node.true_child = Node()
            current_node.true_child.data = Dataset(true_data, true_data_labels)
            current_node.false_child = Node()
            current_node.false_child.data = Dataset(false_data, false_data_labels)

            self._divide(current_node.true_child, depth+1)
            self._divide(current_node.false_child, depth+1)
            # print(split_feature)
        # print(current_node.find_feature_which_best_splits())


if __name__ == '__main__':
    # hard-coded parameters
    max_depth = 10
    fig_output = os.path.join("..", "output", "Decision_Tree_max_depth_10")
    fig_title = 'Decision Tree (Max_Depth=10) Classification [Tyler Trimble]'

    # generate test and training data
    data1 = generate_data(500, [1, 1], [[0.3, 0.2], [0.2, 0.2]], 0)
    data2 = generate_data(500, [3, 4], [[0.3, 0], [0, 0.2]], 1)
    training_dataset = combine_data(data1, data2)
    test_dataset = generate_data(300, [2, 3], [[0.3, 0.0], [0.0, 0.2]], -1)

    # print("Features: " + str(find_features_for_continuous_data(training_dataset)))
    # find features in the training data
    features = find_features_for_continuous_data(training_dataset)
    feature_index = 0
    for dimension_number, dimension_features in enumerate(features):
        for val in dimension_features:
            print("feature " + str(feature_index) + ": is dimension " + str(dimension_number) + " < " + str(val))
            feature_index += 1

    # featurize the training and test dataset
    featurized_training_dataset = featurize_continuous_data(training_dataset, features)
    featurized_test_dataset = featurize_continuous_data(test_dataset, features)

    # create a Decision Tree object
    sample_tree = DecisionTree()
    sample_tree.train(featurized_training_dataset)
    featurized_test_dataset.labels = sample_tree.predict(featurized_test_dataset)

    # split the data by class - for plotting (use the non-featurized data to plot)
    predicted1_samples = test_dataset.samples[(featurized_test_dataset.labels == 0), :]
    predicted2_samples = test_dataset.samples[(featurized_test_dataset.labels == 1), :]

    # plot the samples
    fig = plt.figure()
    plt.plot(data1.samples[:, 0], data1.samples[:, 1], 'b.', label='given class a')
    plt.plot(data2.samples[:, 0], data2.samples[:, 1], 'r.', label='given class b')
    plt.plot(predicted1_samples[:, 0], predicted1_samples[:, 1], 'g*', label='predicted class a')
    plt.plot(predicted2_samples[:, 0], predicted2_samples[:, 1], '*', color='orange', label='predicated class b')
    plt.xlabel('X-axis')
    plt.ylabel('Y-Axis')
    plt.title(fig_title)
    plt.tight_layout()
    plt.grid(True, lw=0.5)
    plt.legend()
    fig.savefig(fig_output)
