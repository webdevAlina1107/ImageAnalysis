import numpy as np
import scipy.stats as stats

from ipl.logging_ import logger

IMAGE_DATA_TYPE = np.uint8


def construct_values_occurrences_map(array: np.ndarray):
    unique_values, counts = np.unique(array, return_counts=True)
    logger.debug('Constructing value occurrences map in matrix')
    return {value: count for value, count in zip(unique_values, counts)}


def calculate_clouds_percentile(array: np.ndarray):
    logger.debug('Calculating clouds percentile')
    boolean_mask = np.logical_or(array == 254, array == 255)
    return np.count_nonzero(boolean_mask) / array.size


def calculate_confidence_interval(array: np.ndarray,
                                  confidence_percent: float = 0.95):
    logger.debug(f'Calculating confidence interval with confidence percent {confidence_percent}')
    array_mean = np.nanmean(array)
    sd = np.sqrt(np.nansum(np.power(array - array_mean, 2)) / array.size - 1)
    alpha = 1 - confidence_percent
    interval = stats.t.ppf(1.0 - (alpha / 2.0), array.size - 1) * (sd / np.sqrt(array.size))
    return array_mean - interval, array_mean + interval


def calculate_all_statistics(array: np.ndarray):
    ci_lower, ci_upper = calculate_confidence_interval(array)
    mean = np.mean(array)
    std = np.std(array)
    cloud_rate = calculate_clouds_percentile(array)
    return cloud_rate, mean, std, ci_lower, ci_upper
