import csv
from argparse import ArgumentParser

import numpy as np
from accelerometerfeatures.utils.pytorch.dataset import \
    AccelerometerDatasetLoader
from scipy.signal import stft
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from smoothing.pwctvdrobust import pwc_tvdrobust

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

    for window_size in [15, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300]:
        print()
        print('###############################################################')
        print(f'Running evaluation for window size {window_size}')
        print()

        step_size = 10
        data_loader = AccelerometerDatasetLoader(
            training_data_file_path, window_size, step_size, True)

        results_file_name = f'w{window_size}_s{step_size}_magnitude_stft.csv'

        with open(results_file_name, 'w') as res_file:
            csv_writer = csv.writer(res_file)
            csv_header = [
                'test_user', 'num_samples', 'algorithm', 'matched', 'missed',
                'acc', 'smoothed_matched', 'smoothed_missed', 'smoothed_acc']
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
                    'k_nearest_neighbors_uniform':
                        KNeighborsClassifier(weights='uniform'),
                    'k_nearest_neighbors_distance':
                        KNeighborsClassifier(weights='distance'),
                    'gradient_boosting': GradientBoostingClassifier(),
                    'extra_trees': ExtraTreesClassifier()

                }
                train_users = data_loader.users[:]
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

                        for frequency_window in stft(magnitude)[2].real.T:
                            train_windows.append(frequency_window)
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

                    # frequency_window = np.fft.fft(magnitude).real
                    test_windows.append(stft(magnitude)[2].real.T)
                    test_labels.append(label)
                print('Finished test data preparation')

                if len(test_windows) == 0:
                    print(f'No data for test user {test_user}. Skipping...')
                    continue

                for classifier_name, classifier in classifiers.items():
                    print(f'Running prediction for classifier '
                          f'{classifier_name}')
                    predictions = []
                    for local_windows in test_windows:
                        local_predictions = classifier.predict(local_windows)
                        unique_modes, mode_counts = \
                            np.unique(local_predictions, return_counts=True)
                        max_idx = mode_counts.argmax()
                        prevalent_mode = unique_modes[max_idx]
                        predictions.append(prevalent_mode)

                    same = sum(np.array(predictions) == test_labels)
                    not_same = sum(np.array(predictions) != test_labels)
                    acc = same / (same + not_same) * 100
                    print(f'o_len: {len(predictions)}')
                    print(f'o_==:  {same}')
                    print(f'o_!=:  {not_same}')
                    print(f'o_acc: {acc}')

                    prediction_ints = [MODE_TO_INT[m] for m in predictions]
                    print('Running smoothing...')
                    smoothed_ints = pwc_tvdrobust(prediction_ints)
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

                    # test_user, num_samples, algorithm, matched, missed, acc,
                    # smoothed_matched, smoothed_missed, smoothed_acc
                    csv_writer.writerow([
                        test_user, len(predictions), classifier_name,
                        same, not_same, acc, smoothed_same,
                        smoothed_not_same, smoothed_acc])


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('training_data_file_path')
    args = argparser.parse_args()

    leave_one_user_out_training(args.training_data_file_path)
