import os
import pickle
from argparse import ArgumentParser
from random import shuffle
from os import listdir
from datetime import timedelta

import numpy as np
import pandas as pd
from accelerometerfeatures.utils.interpolation import Interpolator
from accelerometerfeatures.utils.pytorch.dataset import \
    AccelerometerDatasetLoader
from numpy.fft import rfft
from scipy.constants import g
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from statsmodels.robust import mad

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

MAX_VAL = 2 * g
MIN_VAL = -2 * g
WINDOW_SIZE = 120
STEP_SIZE = 10
# TODO: sync with data loader
BIGGEST_ACCEPTABLE_GAP_IN_SECS = 7
SAMPLE_FREQ = 16


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
    # trimming
    window_data_x = window_data.x[:]
    window_data_y = window_data.y[:]
    window_data_z = window_data.z[:]

    window_data_x[window_data_x > MAX_VAL] = MAX_VAL
    window_data_x[window_data_x < MIN_VAL] = MIN_VAL
    window_data_y[window_data_y > MAX_VAL] = MAX_VAL
    window_data_y[window_data_y < MIN_VAL] = MIN_VAL
    window_data_z[window_data_z > MAX_VAL] = MAX_VAL
    window_data_z[window_data_z < MIN_VAL] = MIN_VAL

    assert np.sum(window_data_x[window_data_x > MAX_VAL]) == 0
    assert np.sum(window_data_x[window_data_x < MIN_VAL]) == 0

    assert np.sum(window_data_y[window_data_y > MAX_VAL]) == 0
    assert np.sum(window_data_y[window_data_y < MIN_VAL]) == 0

    assert np.sum(window_data_z[window_data_z > MAX_VAL]) == 0
    assert np.sum(window_data_z[window_data_z < MIN_VAL]) == 0

    magnitude = np.sqrt(window_data_x**2 + window_data_y**2 + window_data_z**2)

    # min
    x_min = np.min(window_data_x)
    y_min = np.min(window_data_y)
    z_min = np.min(window_data_z)
    overall_min = np.min(magnitude)

    # max
    x_max = np.max(window_data_x)
    y_max = np.max(window_data_y)
    z_max = np.max(window_data_z)
    overall_max = np.max(magnitude)

    # mean
    x_mean = np.mean(window_data_x)
    y_mean = np.mean(window_data_y)
    z_mean = np.mean(window_data_z)
    overall_mean = np.mean(magnitude)

    # standard deviation
    x_stdev = np.std(window_data_x)
    y_stdev = np.std(window_data_y)
    z_stdev = np.std(window_data_z)
    overall_stdev = np.std(magnitude)

    # mean average deviation
    x_mad = mad(window_data_x)
    y_mad = mad(window_data_y)
    z_mad = mad(window_data_z)
    overall_mad = mad(magnitude)

    # skewness
    x_skewness = skew(window_data_x)
    y_skewness = skew(window_data_y)
    z_skewness = skew(window_data_z)
    overall_skewness = skew(magnitude)

    # kurtosis
    x_kurtosis = kurtosis(window_data_x)
    y_kurtosis = kurtosis(window_data_y)
    z_kurtosis = kurtosis(window_data_z)
    overall_kurtosis = kurtosis(magnitude)

    # root mean square amplitude
    x_rms_amplitude = np.sqrt(np.abs(np.mean(window_data_x)))
    y_rms_amplitude = np.sqrt(np.abs(np.mean(window_data_y)))
    z_rms_amplitude = np.sqrt(np.abs(np.mean(window_data_z)))
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
    x_window_energy = np.sum(window_data_x)
    y_window_energy = np.sum(window_data_y)
    z_window_energy = np.sum(window_data_z)
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
    x_spectral_centroid = spectral_centroid(window_data_x)
    y_spectral_centroid = spectral_centroid(window_data_y)
    z_spectral_centroid = spectral_centroid(window_data_z)
    overall_spectral_centroid = spectral_centroid(magnitude)

    # spectral energy
    x_spectral_energy = np.sum(np.fft.fft(window_data_x).real)
    y_spectral_energy = np.sum(np.fft.fft(window_data_y).real)
    z_spectral_energy = np.sum(np.fft.fft(window_data_z).real)
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


def train(training_data_file_path):
    data_loader = AccelerometerDatasetLoader(
        training_data_file_path, WINDOW_SIZE, STEP_SIZE, True)

    classifiers = {
        # 'decision_tree': DecisionTreeClassifier(),
        'random_forrest': RandomForestClassifier(),
        # 'svm': SVC(
        #     gamma='scale',
        #     decision_function_shape='ovo',
        #     probability=True),
        # 'adaboost': AdaBoostClassifier(),
        # 'radius_neighbors_uniform':
        #     RadiusNeighborsClassifier(weights='uniform'),
        # 'radius_neighbors_distance':
        #     RadiusNeighborsClassifier(weights='distance'),
        # 'k_nearest_neighbors_uniform':
        #     KNeighborsClassifier(weights='uniform'),
        # 'k_nearest_neighbors_distance':
        #     KNeighborsClassifier(weights='distance'),
        'gradient_boosting': GradientBoostingClassifier(),
        'extra_trees': ExtraTreesClassifier()

    }
    train_users = data_loader.users

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

    for classifier_name, classifier in classifiers.items():
        print(f'Training classifier {classifier_name}')
        # classifier.fit(train_windows, train_labels)
        classifier.fit(cleaned_train_windows, cleaned_train_labels)
        with open(f'{classifier_name}.pickle', 'wb') as classifier_file:
            pickle.dump(classifier, classifier_file)
    print('Finished training')


def predict(input_data_file_path, model_dir_path, output_file_path=None):
    majority_smoother = MajorityVoteSmoother(10, 15)

    df = pd.read_csv(input_data_file_path, parse_dates=[1])
    df.timestamp = pd.to_datetime(df.timestamp)

    interpolator = Interpolator(df, SAMPLE_FREQ, STEP_SIZE)
    expected_no_samples_per_window = WINDOW_SIZE * SAMPLE_FREQ

    interpolated_sub_datasets = [
        sub_ds
        for sub_ds
        in interpolator.get_interpolated_data()
        if len(sub_ds) >= expected_no_samples_per_window]

    sub_dataset_windows = []

    for data_shred in interpolated_sub_datasets:
        windows = []
        first_idx = data_shred.first_valid_index()
        last_idx = data_shred.last_valid_index()
        last_datetime = data_shred.timestamp[last_idx]

        start_datetime = data_shred.timestamp[first_idx]
        end_datetime = start_datetime + timedelta(seconds=WINDOW_SIZE)

        while end_datetime <= last_datetime:
            win_idxs = np.logical_and(
                data_shred.timestamp >= start_datetime,
                data_shred.timestamp < end_datetime)

            window_data = data_shred[win_idxs]
            window_data.reset_index(drop=True, inplace=True)

            # for the next round
            start_datetime = start_datetime + timedelta(seconds=STEP_SIZE)
            end_datetime = start_datetime + timedelta(seconds=WINDOW_SIZE)

            if len(window_data) < expected_no_samples_per_window:
                continue
            else:
                assert len(window_data) == \
                       expected_no_samples_per_window

            windows.append(window_data)
        sub_dataset_windows.append(windows)

    models = []

    for model_file_name in listdir(model_dir_path):
        if not model_file_name.endswith('.pickle'):
            continue

        with open(os.path.join(model_dir_path, model_file_name), 'br') as model_file:
            models.append(pickle.load(model_file))

    if len(sub_dataset_windows) == 0:
        # write dummy output
        with open(output_file_path, 'w') as out_file:
            out_file.writelines([])
        return

    stages = {}
    for windows in sub_dataset_windows:
        for model in models:
            model_name = str(model).replace('\n', ' ').replace(' ', '')
            if stages.get(model_name) is None:
                stages[model_name] = []

            feature_windows = [prepare_features(w) for w in windows]
            probabilities = model.predict_proba(feature_windows)
            predictions = model.predict(feature_windows)
            smoothed_predictions = majority_smoother.smooth(predictions)

            if len(predictions) == 1:
                lbl = predictions[0]
                lbl_idx = np.argmax(model.classes_ == lbl)
                stages[model_name].append((
                    smoothed_predictions[0],
                    windows[0].timestamp[0],
                    windows[0].timestamp[0],
                    probabilities[0, lbl_idx]))

            else:
                start_idx = 0
                for idx in range(len(predictions)-1):
                    if not smoothed_predictions[idx] == smoothed_predictions[idx+1]:
                        end_idx = idx
                        lbl = smoothed_predictions[start_idx]
                        lbl_idx = np.argmax(model.classes_ == lbl)
                        avg_probability = np.mean(
                            probabilities[start_idx: end_idx, lbl_idx])

                        stages[model_name].append((
                            lbl, windows[start_idx].timestamp[0],
                            windows[end_idx].timestamp[0],
                            avg_probability))
                        start_idx = idx + 1

                lbl = smoothed_predictions[start_idx]
                lbl_idx = np.argmax(model.classes_ == lbl)
                avg_probability = np.mean(
                    probabilities[start_idx: len(predictions)-1, lbl_idx])
                stages[model_name].append((
                    smoothed_predictions[start_idx],
                    windows[start_idx].timestamp[0],
                    windows[len(predictions)-1].timestamp[0],
                    avg_probability))

    model_names = list(stages.keys())
    overall_avg_probabilities = []

    # chose model with highest average prediction probability
    for model_name in model_names:
        model_stages = stages[model_name]

        if len(model_stages) == 0:
            # write dummy output and return
            with open(output_file_path, 'w') as out_file:
                out_file.writelines([])
            return

        overall_avg_probabilities.append(
            np.mean([prob for _, _, _, prob in model_stages]))

    best_model_idx = np.argmax(overall_avg_probabilities)
    best_model = model_names[best_model_idx]
    best_predictions = stages[best_model]

    if output_file_path is None:
        for stage in best_predictions:
            print('%s,%s,%s,%f,%s' % (
                stage[0], stage[1], stage[2], stage[3], best_model))
    else:
        with open(output_file_path, 'w') as out_file:
            out_file.writelines(
                ['%s,%s,%s,%f,%s' % (stg[0], stg[1], stg[2], stg[3], best_model)
                 for stg in best_predictions])


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('input_data_file_path')
    argparser.add_argument('--train', action="store_true", default=False)
    argparser.add_argument('--predict', action="store_true", default=False)
    argparser.add_argument('--model_dir')
    argparser.add_argument('--output_file')
    args = argparser.parse_args()

    if args.train:
        train(args.input_data_file_path)
    elif args.predict:
        predict(args.input_data_file_path, args.model_dir, args.output_file)
