from pathlib import Path

import numpy as np
import scipy.stats as stats

from ipl.logging_ import logger

IMAGE_DATA_TYPE = np.uint8
SPECIAL_VALUE: IMAGE_DATA_TYPE = IMAGE_DATA_TYPE(255)


def fill_cloud_bits_with_value(array: np.ndarray,
                               clouds_special_value: IMAGE_DATA_TYPE = SPECIAL_VALUE
                               ) -> np.ndarray:
    non_clouds_range = (1, 253)
    masked_array = np.ma.masked_outside(array, *non_clouds_range)
    logger.debug(f'Filling image bits out of range {non_clouds_range} with {clouds_special_value}')
    # Not np.nan because it's only for arrays with float data type only
    return masked_array.filled(clouds_special_value)


def construct_values_occurrences_map(array: np.ndarray):
    unique_values, counts = np.unique(array, return_counts=True)
    logger.debug('Constructing value occurrences map in matrix')
    return {value: count for value, count in zip(unique_values, counts)}


def calculate_clouds_percentile(array: np.ndarray,
                                clouds_special_value: IMAGE_DATA_TYPE = SPECIAL_VALUE):
    """Note that this function should be only used after processing with fill_cloud_bits_with_value"""
    logger.debug('Calculating clouds percentile')
    if clouds_special_value is not np.nan:
        boolean_mask = array == clouds_special_value
    else:
        boolean_mask = np.isnan(array)

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
    array = fill_cloud_bits_with_value(array)
    cloud_rate = calculate_clouds_percentile(array)
    return cloud_rate, mean, std, ci_lower, ci_upper
