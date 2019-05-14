import csv
from argparse import ArgumentParser

import numpy as np
from accelerometerfeatures.utils.pytorch.dataset import \
    AccelerometerDatasetLoader
from numpy.fft import rfft
from scipy.stats import entropy, skew, kurtosis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.robust import mad

from smoothing import pwctvdrobust
from smoothing.evaluation import MajorityVoteSmoother

from random import shuffle

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


def spectral_centroid(signal):
    """
    spectral centroid
    calculation taken from
    https://gist.github.com/endolith/359724/aa7fcc043776f16f126a0ccd12b599499509c3cc
    """
    spectrum = np.abs(rfft(signal))
    if np.sum(spectrum) == 0:
        normalized_spectrum = spectrum
    else:
        # like a probability mass function
        normalized_spectrum = spectrum / np.sum(spectrum)
    normalized_frequencies = np.linspace(0, 1, len(spectrum))
    spectral_centroid_val = np.sum(normalized_frequencies * normalized_spectrum)

    return spectral_centroid_val


def prepare_features(window_data):
    magnitude = np.sqrt(window_data.x**2 + window_data.y**2 + window_data.z**2)

    # min
    x_min = np.min(window_data.x)
    y_min = np.min(window_data.y)
    z_min = np.min(window_data.z)
    overall_min = np.min(magnitude)

    # max
    x_max = np.max(window_data.x)
    y_max = np.max(window_data.y)
    z_max = np.max(window_data.z)
    overall_max = np.max(magnitude)

    # mean
    x_mean = np.mean(window_data.x)
    y_mean = np.mean(window_data.y)
    z_mean = np.mean(window_data.z)
    overall_mean = np.mean(magnitude)

    # standard deviation
    x_stdev = np.std(window_data.x)
    y_stdev = np.std(window_data.y)
    z_stdev = np.std(window_data.z)
    overall_stdev = np.std(magnitude)

    # mean average deviation
    x_mad = mad(window_data.x)
    y_mad = mad(window_data.y)
    z_mad = mad(window_data.z)
    overall_mad = mad(magnitude)

    # skewness
    x_skewness = skew(window_data.x)
    y_skewness = skew(window_data.y)
    z_skewness = skew(window_data.z)
    overall_skewness = skew(magnitude)

    # kurtosis
    x_kurtosis = kurtosis(window_data.x)
    y_kurtosis = kurtosis(window_data.y)
    z_kurtosis = kurtosis(window_data.z)
    overall_kurtosis = kurtosis(magnitude)

    # root mean square amplitude
    x_rms_amplitude = np.sqrt(np.abs(np.mean(window_data.x)))
    y_rms_amplitude = np.sqrt(np.abs(np.mean(window_data.y)))
    z_rms_amplitude = np.sqrt(np.abs(np.mean(window_data.z)))
    overall_rms_amplitude = np.sqrt(np.abs(np.mean(magnitude)))

    covariance_matrix = np.cov(window_data[['x', 'y', 'z']])

    # covariance of two values
    x_y_covariance = covariance_matrix[0, 1]
    x_z_covariance = covariance_matrix[0, 2]
    y_z_covariance = covariance_matrix[1, 2]

    # min covariance of two values
    min_covariance = np.min([x_y_covariance, x_z_covariance, y_z_covariance])

    # max covariance of two values
    max_covariance = np.max([x_y_covariance, x_z_covariance, y_z_covariance])

    # window energy
    x_window_energy = np.sum(window_data.x)
    y_window_energy = np.sum(window_data.y)
    z_window_energy = np.sum(window_data.z)
    overall_window_energy = np.sum(magnitude)

    # window entropy
    # x_window_entropy = entropy(window_data.x)
    # y_window_entropy = entropy(window_data.y)
    # z_window_entropy = entropy(window_data.z)

    # min_window_entropy = np.min([
    #     x_window_entropy, y_window_entropy, z_window_entropy])
    # max_window_entropy = np.max([
    #     x_window_entropy, y_window_entropy, z_window_entropy])
    # overall_window_entropy = entropy(magnitude)

    # Fourier transform
    frequency_component_amplitudes = np.fft.fft(magnitude).real

    # spectral centroid
    x_spectral_centroid = spectral_centroid(window_data.x)
    y_spectral_centroid = spectral_centroid(window_data.y)
    z_spectral_centroid = spectral_centroid(window_data.z)
    overall_spectral_centroid = spectral_centroid(magnitude)

    # spectral energy
    x_spectral_energy = np.sum(np.fft.fft(window_data.x).real)
    y_spectral_energy = np.sum(np.fft.fft(window_data.y).real)
    z_spectral_energy = np.sum(np.fft.fft(window_data.z).real)
    overall_spectral_energy = np.sum(frequency_component_amplitudes)

    # spectral entropy
    # x_spectral_entropy = entropy(np.fft.fft(window_data.x).real)
    # y_spectral_entropy = entropy(np.fft.fft(window_data.y).real)
    # z_spectral_entropy = entropy(np.fft.fft(window_data.z).real)
    # overall_spectral_entropy = entropy(frequency_component_amplitudes)

    features = (
        x_min, y_min, z_min, overall_min,
        x_max, y_max, z_max, overall_max,
        x_mean, y_mean, z_mean, overall_mean,
        x_stdev, y_stdev, z_stdev, overall_stdev,
        x_mad, y_mad, z_mad, overall_mad,
        x_skewness, y_skewness, z_skewness, overall_skewness,
        x_kurtosis, y_kurtosis, z_kurtosis, overall_kurtosis,
        x_rms_amplitude, y_rms_amplitude, z_rms_amplitude,
        overall_rms_amplitude,
        x_y_covariance, x_z_covariance, y_z_covariance,
        min_covariance, max_covariance,
        x_window_energy, y_window_energy, z_window_energy,
        overall_window_energy,
        # x_window_entropy, y_window_entropy, z_window_entropy,
        # min_window_entropy, max_window_entropy,
        # overall_window_entropy,
        x_spectral_centroid, y_spectral_centroid, z_spectral_centroid,
        overall_spectral_centroid,
        x_spectral_energy, y_spectral_energy, z_spectral_energy,
        overall_spectral_energy,
        # x_spectral_entropy, y_spectral_entropy, z_spectral_entropy,
        # overall_spectral_entropy
    )

    features += tuple(frequency_component_amplitudes)
    return features


def leave_one_user_out_training(training_data_file_path):

    for window_size in [1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300]:
        print()
        print('###############################################################')
        print(f'Running evaluation for window size {window_size}')
        print()

        step_size = 10
        majority_smoother = MajorityVoteSmoother(10, 15)
        data_loader = AccelerometerDatasetLoader(
            training_data_file_path, window_size, step_size, True)

        results_file_name = \
            f'w{window_size}_s{step_size}__all_features__w_interpolation_' \
            f'w_maj_vote__w_pvcrobust__shuffled__1st_2nd_3rd.csv'

        with open(results_file_name, 'w') as res_file:
            csv_writer = csv.writer(res_file)
            csv_header = [
                'test_user', 'num_samples', 'algorithm', 'avg_window_entropy',
                'matched', 'missed', 'acc', 'smoothed_matched',
                'smoothed_missed', 'smoothed_acc', 'maj_vote_smoothed_acc']
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
                    for window_data, label in \
                            data_loader.get_user_data_windows(train_user):

                        features = prepare_features(window_data)
                        train_windows.append(features)
                        train_labels.append(label)
                print('Finished train data preparation')

                train_data = list(zip(train_windows, train_labels))
                shuffle(train_data)

                # clean up
                cleaned_train_windows = []
                cleaned_train_labels = []
                for i in range(len(train_data)):
                    train_window = train_data[i][0]

                    if sum(np.isfinite(train_window)) == len(train_window):
                        cleaned_train_windows.append(train_window)
                        cleaned_train_labels.append(train_data[i][1])

                if len(cleaned_train_windows) == 0:
                    print('No train data left after cleaning...')
                    import pdb; pdb.set_trace()
                    continue

                for classifier_name, classifier in classifiers.items():
                    print(f'Training classifier {classifier_name}')
                    classifier.fit(cleaned_train_windows, cleaned_train_labels)
                print('Finished training')

                test_windows = []
                test_labels = []

                print(f'Preparing test data for user {test_user}')
                for window_data, label in \
                        data_loader.get_user_data_windows(test_user):
                    features = prepare_features(window_data)

                    test_windows.append(features)
                    test_labels.append(label)
                print('Finished test data preparation')

                if len(test_windows) == 0:
                    print(f'No data for test user {test_user}. Skipping...')
                    continue

                # clean up
                cleaned_test_windows = []
                cleaned_test_labels = []
                for i in range(len(test_windows)):
                    test_window = test_windows[i]

                    if sum(np.isfinite(test_window)) == len(test_window):
                        cleaned_test_windows.append(test_window)
                        cleaned_test_labels.append(test_labels[i])

                if len(cleaned_test_windows) == 0:
                    print('No test data left after cleaning...')
                    continue

                for classifier_name, classifier in classifiers.items():
                    print(f'Running prediction for classifier '
                          f'{classifier_name}')
                    predictions = classifier.predict(cleaned_test_windows)

                    same = sum(predictions == cleaned_test_labels)
                    not_same = sum(predictions != cleaned_test_labels)
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
                    smoothed_same = \
                        sum(smoothed_predictions == cleaned_test_labels)
                    smoothed_not_same = \
                        sum(smoothed_predictions != cleaned_test_labels)
                    smoothed_acc = \
                        smoothed_same / len(cleaned_test_labels) * 100

                    print(f's_len: {len(predictions)}')
                    print(f's_==:  {smoothed_same}')
                    print(f's_!=:  {smoothed_not_same}')
                    print(f's_acc: {smoothed_acc}')

                    majority_smoothed_same = \
                        sum(majority_smoothed == cleaned_test_labels)
                    majority_smoothed_not_same = \
                        sum(majority_smoothed != cleaned_test_labels)
                    majority_smoothed_acc = \
                        majority_smoothed_same / len(cleaned_test_labels) * 100
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
