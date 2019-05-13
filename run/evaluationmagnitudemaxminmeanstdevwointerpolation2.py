import csv
from argparse import ArgumentParser

import numpy as np
from accelerometerfeatures.utils.pytorch.dataset import \
    AccelerometerDatasetLoader
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from smoothing import pwctvdrobust
from smoothing.evaluation import MajorityVoteSmoother

MODE_TO_INT = {
    'walk': 1,
    'bike': 2,
    'e-bike': 3,
    'car': 4,
    'bus': 5,
    'train': 6,
}

INT_TO_MODE = {
    1: 'walk',
    2: 'bike',
    3: 'e-bike',
    4: 'car',
    5: 'bus',
    6: 'train',
}


def leave_one_user_out_training(training_data_file_path):

    for window_size in [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90]:
        print()
        print('###############################################################')
        print(f'Running evaluation for window size {window_size}')
        print()

        step_size = 10
        majority_smoother = MajorityVoteSmoother(10, 15)
        data_loader = AccelerometerDatasetLoader(
            training_data_file_path, window_size, step_size, False)

        results_file_name = \
            f'w{window_size}_s{step_size}_unibo_magnitude_max_min_mean_stdev_' \
            f'w_maj_vote_no_interpolation.csv'

        with open(results_file_name, 'w') as res_file:
            csv_writer = csv.writer(res_file)
            csv_header = [
                'test_user', 'num_samples', 'algorithm', 'matched', 'missed',
                'acc', 'smoothed_matched', 'smoothed_missed', 'smoothed_acc',
                'maj_vote_smoothed_acc']
            csv_writer.writerow(csv_header)

            for user in data_loader.users:
                classifiers = {
                    'decision_tree': DecisionTreeClassifier(),
                    'random_forrest': RandomForestClassifier(),
                    'svm': SVC(
                        gamma='scale',
                        decision_function_shape='ovo',
                        probability=True),
                    'adaboost': AdaBoostClassifier(),
                    'gradient_boosting': GradientBoostingClassifier(),
                    'extra_trees': ExtraTreesClassifier()

                }
                train_users = data_loader.users.copy()
                train_users.remove(user)
                test_user = user

                train_windows = []
                train_labels = []

                for train_user in train_users:
                    print(f'Preparing training data for user {train_user}')
                    for window, label in \
                            data_loader.get_user_data_windows(train_user):

                        magnitude = \
                            np.sqrt(window.x**2 + window.y**2 + window.z**2)

                        maximum = np.max(magnitude)
                        minimum = np.min(magnitude)
                        mean = np.mean(magnitude)
                        stdev = np.std(magnitude)
                        train_windows.append((minimum, maximum, mean, stdev))
                        train_labels.append(label)
                print('Finished train data preparation')

                for classifier_name, classifier in classifiers.items():
                    print(f'Training classifier {classifier_name}')
                    classifier.fit(train_windows, train_labels)
                print('Finished training')

                test_windows = []
                test_labels = []

                print(f'Preparing test data for user {test_user}')
                for window, label in \
                        data_loader.get_user_data_windows(test_user):

                    magnitude = np.sqrt(window.x**2 + window.y**2 + window.z**2)

                    maximum = np.max(magnitude)
                    minimum = np.min(magnitude)
                    mean = np.mean(magnitude)
                    stdev = np.std(magnitude)
                    test_windows.append((minimum, maximum, mean, stdev))
                    test_labels.append(label)
                print('Finished test data preparation')

                if len(test_windows) == 0:
                    print(f'No data for test user {test_user}. Skipping...')
                    continue

                for classifier_name, classifier in classifiers.items():
                    print(f'Running prediction for classifier '
                          f'{classifier_name}')
                    predictions = classifier.predict(test_windows)

                    same = sum(predictions == test_labels)
                    not_same = sum(predictions != test_labels)
                    acc = same / (same + not_same) * 100
                    print(f'o_len: {len(predictions)}')
                    print(f'o_==:  {same}')
                    print(f'o_!=:  {not_same}')
                    print(f'o_acc: {acc}')

                    prediction_ints = [MODE_TO_INT[m] for m in predictions]
                    print('Running smoothing...')
                    smoothed_ints = pwctvdrobust.pwc_tvdrobust(prediction_ints)
                    majority_smoothed = \
                        np.array(majority_smoother.smooth(predictions))
                    print('Smoothing done...')

                    smoothed_predictions = np.array(
                        [INT_TO_MODE[int(round(f))] for f in smoothed_ints])
                    smoothed_same = sum(smoothed_predictions == test_labels)
                    smoothed_not_same = sum(smoothed_predictions != test_labels)
                    smoothed_acc = smoothed_same / len(test_labels) * 100

                    print(f's_len: {len(predictions)}')
                    print(f's_==:  {smoothed_same}')
                    print(f's_!=:  {smoothed_not_same}')
                    print(f's_acc: {smoothed_acc}')

                    majority_smoothed_same = \
                        sum(majority_smoothed == test_labels)
                    majority_smoothed_not_same = \
                        sum(majority_smoothed != test_labels)
                    majority_smoothed_acc = \
                        majority_smoothed_same / len(test_labels) * 100
                    print(f'm_len: {len(predictions)}')
                    print(f'm_==:  {majority_smoothed_same}')
                    print(f'm_!=:  {majority_smoothed_not_same}')
                    print(f'm_acc: {majority_smoothed_acc}')

                    # test_user, num_samples, algorithm, matched, missed, acc,
                    # smoothed_matched, smoothed_missed, smoothed_acc
                    csv_writer.writerow([
                        test_user, len(predictions), classifier_name, same,
                        not_same, acc, smoothed_same, smoothed_not_same,
                        smoothed_acc, majority_smoothed_acc])


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('training_data_file_path')
    args = argparser.parse_args()

    leave_one_user_out_training(args.training_data_file_path)
