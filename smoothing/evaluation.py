import csv
import random
from math import floor

import numpy as np

from smoothing import MajorityVoteSmoother
from smoothing.pwcbilateral import pwc_bilateral
from smoothing.pwccluster import pwc_cluster
from smoothing.pwcjumppenalty import pwc_jumppenalty
from smoothing.pwcmedfiltit import pwc_medfiltit
from smoothing.pwctvdip import pwc_tvdip
from smoothing.pwctvdrobust import pwc_tvdrobust

BIKE = 'bike_'
BUS = 'bus__'
CAR = 'car__'
E_BIKE = 'ebike'
TRAIN = 'train'
WALK = 'walk_'
MODES = [BIKE, BUS, CAR, E_BIKE, TRAIN, WALK]

MODE_TO_INT = {
    WALK: 1,
    BIKE: 2,
    E_BIKE: 3,
    CAR: 4,
    BUS: 5,
    TRAIN: 6,
}

INT_TO_MODE = {
    1: WALK,
    2: BIKE,
    3: E_BIKE,
    4: CAR,
    5: BUS,
    6: TRAIN,
}

WINDOWS_PER_MINUTE = 6

# Idea: Trip with
# - about 5 minutes walk to train station
# - about 30 minutes of train ride
# - about 5 minutes of walk from the station to a bus stop
# - about 15 minutes bus ride
# - about 2 minutes walk to the work place
MODE_LABELS = np.array(
    round(5 * WINDOWS_PER_MINUTE) * [WALK] +
    round(30 * WINDOWS_PER_MINUTE) * [TRAIN] +
    round(5 * WINDOWS_PER_MINUTE) * [WALK] +
    round(15 * WINDOWS_PER_MINUTE) * [BUS] +
    round(2 * WINDOWS_PER_MINUTE) * [WALK], dtype=np.str_)


def randomize(labels, random_percentage):
    num_labels = len(labels)
    num_labels_randomized = floor(random_percentage / 100 * num_labels)
    indexes_to_be_randomized = random.choices(
        [i for i in range(num_labels)], k=num_labels_randomized)

    randomized_labels = labels.copy()

    for idx in indexes_to_be_randomized:
        tmp_modes = MODES.copy()
        tmp_modes.remove(labels[idx])
        new_label = random.choice(tmp_modes)
        randomized_labels[idx] = new_label

    return randomized_labels


def update_results(results, path, value):
    head, *rest = path
    if results.get(head) is None:
        results[head] = {}

    if not rest:
        results[head] = value

        return True
    else:
        return update_results(results[head], rest, value)


def run_majority_vote_evaluation(predictions, window_size, num_iterations):
    majority_vote_smoother = MajorityVoteSmoother(num_iterations, window_size)
    smoothed_modes = majority_vote_smoother.smooth(predictions)

    matched = sum(smoothed_modes == MODE_LABELS)
    not_matched = sum(smoothed_modes != MODE_LABELS)
    acc = matched / (matched + not_matched)

    return acc


def run_pwc_bilateral_soft_kernel(predictions, window_size):
    smoothed_modes_floats = pwc_bilateral(
        y=[MODE_TO_INT[p] for p in predictions],
        soft=True,
        width=window_size)

    smoothed_modes = np.array(
        [INT_TO_MODE[int(round(f))] for f in smoothed_modes_floats],
        dtype=np.str_)

    matched = sum(smoothed_modes == MODE_LABELS)
    not_matched = sum(smoothed_modes != MODE_LABELS)
    acc = matched / (matched + not_matched)

    return acc


def run_pwc_bilateral_hard_kernel(predictions, window_size):
    smoothed_modes_floats = pwc_bilateral(
        y=[MODE_TO_INT[p] for p in predictions],
        soft=False,
        width=window_size)

    smoothed_modes = np.array(
        [INT_TO_MODE[int(round(f))] for f in smoothed_modes_floats],
        dtype=np.str_)

    matched = sum(smoothed_modes == MODE_LABELS)
    not_matched = sum(smoothed_modes != MODE_LABELS)
    acc = matched / (matched + not_matched)

    return acc


def run_pwc_cluster_soft_biased(predictions):
    smoothed_modes_floats = pwc_cluster(
        y=[MODE_TO_INT[p] for p in predictions],
        soft=True,
        biased=True)

    smoothed_modes = np.array(
        [INT_TO_MODE[int(round(f))] for f in smoothed_modes_floats],
        dtype=np.str_)

    matched = sum(smoothed_modes == MODE_LABELS)
    not_matched = sum(smoothed_modes != MODE_LABELS)
    acc = matched / (matched + not_matched)

    return acc


def run_pwc_cluster_hard_biased(predictions):
    smoothed_modes_floats = pwc_cluster(
        y=[MODE_TO_INT[p] for p in predictions],
        soft=False,
        biased=True)

    smoothed_modes = np.array(
        [INT_TO_MODE[int(round(f))] for f in smoothed_modes_floats],
        dtype=np.str_)

    matched = sum(smoothed_modes == MODE_LABELS)
    not_matched = sum(smoothed_modes != MODE_LABELS)
    acc = matched / (matched + not_matched)

    return acc


def run_pwc_cluster_soft_unbiased(predictions):
    smoothed_modes_floats = pwc_cluster(
        y=[MODE_TO_INT[p] for p in predictions],
        soft=True,
        biased=False)

    smoothed_modes = np.array(
        [INT_TO_MODE[int(round(f))] for f in smoothed_modes_floats],
        dtype=np.str_)

    matched = sum(smoothed_modes == MODE_LABELS)
    not_matched = sum(smoothed_modes != MODE_LABELS)
    acc = matched / (matched + not_matched)

    return acc


def run_pwc_cluster_hard_unbiased(predictions):
    smoothed_modes_floats = pwc_cluster(
        y=[MODE_TO_INT[p] for p in predictions],
        soft=False,
        biased=False)

    smoothed_modes = np.array(
        [INT_TO_MODE[int(round(f))] for f in smoothed_modes_floats],
        dtype=np.str_)

    matched = sum(smoothed_modes == MODE_LABELS)
    not_matched = sum(smoothed_modes != MODE_LABELS)
    acc = matched / (matched + not_matched)

    return acc


def run_pwc_jump_penalty_least_sqares(predictions, gamma):
    smoothed_modes_floats = pwc_jumppenalty(
        y=[MODE_TO_INT[p] for p in predictions],
        square=True,
        gamma=gamma)

    smoothed_modes = np.array(
        [INT_TO_MODE[int(round(f))] for f in smoothed_modes_floats],
        dtype=np.str_)

    matched = sum(smoothed_modes == MODE_LABELS)
    not_matched = sum(smoothed_modes != MODE_LABELS)
    acc = matched / (matched + not_matched)

    return acc


def run_pwc_jump_penalty_least_absolutes(predictions, gamma):
    smoothed_modes_floats = pwc_jumppenalty(
        y=[MODE_TO_INT[p] for p in predictions],
        square=False,
        gamma=gamma)

    smoothed_modes = np.array(
        [INT_TO_MODE[int(round(f))] for f in smoothed_modes_floats],
        dtype=np.str_)

    matched = sum(smoothed_modes == MODE_LABELS)
    not_matched = sum(smoothed_modes != MODE_LABELS)
    acc = matched / (matched + not_matched)

    return acc


def run_median_filtering(predictions, window_size):
    # Each element of kernel_size should be odd.
    if window_size % 2 == 0:
        win_size = window_size + 1
    else:
        win_size = window_size

    smoothed_modes_floats = pwc_medfiltit(
        y=[MODE_TO_INT[p] for p in predictions],
        W=win_size)

    smoothed_modes = np.array(
        [INT_TO_MODE[int(round(f))] for f in smoothed_modes_floats],
        dtype=np.str_)

    matched = sum(smoothed_modes == MODE_LABELS)
    not_matched = sum(smoothed_modes != MODE_LABELS)
    acc = matched / (matched + not_matched)

    return acc


def run_pwc_tvdip(predictions):
    smoothed_modes_floats = pwc_tvdip(y=[MODE_TO_INT[p] for p in predictions])
    smoothed_modes = np.array(
        [INT_TO_MODE[int(round(f[0]))] for f in smoothed_modes_floats.T],
        dtype=np.str_)

    matched = sum(smoothed_modes == MODE_LABELS)
    not_matched = sum(smoothed_modes != MODE_LABELS)
    acc = matched / (matched + not_matched)

    return acc


def run_pwc_tvdrobust(predictions, lambda_):
    smoothed_modes_floats = pwc_tvdrobust(
        y=[MODE_TO_INT[p] for p in predictions],
        lamb=lambda_)

    smoothed_modes = np.array(
        [INT_TO_MODE[int(round(f))] for f in smoothed_modes_floats],
        dtype=np.str_)

    matched = sum(smoothed_modes == MODE_LABELS)
    not_matched = sum(smoothed_modes != MODE_LABELS)
    acc = matched / (matched + not_matched)

    return acc


def get_majority_vote_data(results: dict):
    xs = []
    values_and_labels = {}

    for perc_randomized in results.keys():
        xs.append(perc_randomized)
        majority_vote_data = results[perc_randomized]['majority_vote']

        for window_size in majority_vote_data.keys():
            win_majority_vote_data = majority_vote_data[window_size]
            for num_iterations in win_majority_vote_data.keys():
                this_num_iter_data = win_majority_vote_data[num_iterations]

                acc_values = []
                for trial_num, entry in this_num_iter_data.items():
                    acc_values.append(entry['acc'])

                avg_acc = np.average(acc_values)
                label = f'majority vote, w{window_size}, iter{num_iterations}'
                if values_and_labels.get(label) is None:
                    values_and_labels[label] = []
                values_and_labels[label].append(avg_acc)

    return xs, values_and_labels


def main():
    csv_writer = csv.writer(open('res.csv', 'w'))
    csv_writer.writerow(['approach', 'acc', 'perc_randomized', 'trial_num'])

    for trial_num in range(50):
        for perc_randomized in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            predictions = randomize(MODE_LABELS, perc_randomized)

            for window_size in range(3, 21):
                for num_iterations in [5, 10, 20, 30, 40, 50]:
                    acc = run_majority_vote_evaluation(
                        predictions, window_size, num_iterations)

                    csv_writer.writerow([
                        'majority_vote_w%i_i%i' % (window_size, num_iterations),
                        acc,
                        perc_randomized,
                        trial_num])

                acc = run_pwc_bilateral_soft_kernel(predictions, window_size)
                csv_writer.writerow([
                    'pwc_bilateral_soft_kernel_w%i' % window_size,
                    acc,
                    perc_randomized,
                    trial_num])

                acc = run_pwc_bilateral_hard_kernel(predictions, window_size)
                csv_writer.writerow([
                    'pwc_bilateral_hard_kernel_w%i' % window_size,
                    acc,
                    perc_randomized,
                    trial_num])

                acc = run_median_filtering(predictions, window_size)
                csv_writer.writerow([
                    'median_filtering_w%i' % window_size,
                    acc,
                    perc_randomized,
                    trial_num])

            acc = run_pwc_cluster_soft_biased(predictions)
            csv_writer.writerow(
                ['pwc_cluster_soft_biased', acc, perc_randomized, trial_num])

            acc = run_pwc_cluster_hard_biased(predictions)
            csv_writer.writerow(
                ['pwc_cluster_hard_biased', acc, perc_randomized, trial_num])

            acc = run_pwc_cluster_soft_unbiased(predictions)
            csv_writer.writerow(
                ['pwc_cluster_soft_unbiased', acc, perc_randomized, trial_num])

            acc = run_pwc_cluster_hard_unbiased(predictions)
            csv_writer.writerow(
                ['pwc_cluster_hard_unbiased', acc, perc_randomized, trial_num])

            acc = run_pwc_tvdip(predictions)
            csv_writer.writerow(['pwc_tvdip', acc, perc_randomized, trial_num])

            for gamma in [0.1, 0.5, 1, 2, 5, 10]:
                acc = run_pwc_jump_penalty_least_sqares(predictions, gamma)
                csv_writer.writerow([
                    'pwc_jump_penalty_least_squares_g%f' % gamma,
                    acc,
                    perc_randomized,
                    trial_num])

                acc = run_pwc_jump_penalty_least_absolutes(predictions, gamma)
                csv_writer.writerow([
                    'pwc_jump_penalty_least_absolutes_g%f' % gamma,
                    acc,
                    perc_randomized,
                    trial_num])

            for lambda_ in [0.1, 0.5, 1, 2, 5, 10, 20, 50]:
                acc = run_pwc_tvdrobust(predictions, lambda_)
                csv_writer.writerow([
                    'pwc_tvdrobust_l%f' % lambda_,
                    acc,
                    perc_randomized,
                    trial_num])


if __name__ == '__main__':
    main()
