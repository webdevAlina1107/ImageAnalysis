from pathlib import Path

import numpy as np
import rasterio as rast
import scipy.stats as stats

from ipl._logging import logger

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


def read_image_bitmap(image_file_path: Path,
                      selected_band: int = 1
                      ) -> np.ndarray:
    logger.debug(f'Reading image at {image_file_path}, band = {selected_band}')
    with rast.open(image_file_path, 'r', dtype=IMAGE_DATA_TYPE) as raster:
        return raster.read(selected_band)


def construct_values_occurrences_map(array: np.ndarray):
    unique_values, counts = np.unique(array, return_counts=True)
    logger.debug('Constructing value occurrences map in matrix')
    return {value: count for value, count in zip(unique_values, counts)}


def calculate_clouds_percentile(array: np.ndarray,
                                clouds_special_value: IMAGE_DATA_TYPE = SPECIAL_VALUE):
    """Note that this function should be only used after processing with fill_cloud_bits_with_value"""
    logger.debug('Calculating clouds percentile')
    return np.count_nonzero(array == clouds_special_value) / array.size


def calculate_confidence_interval(array: np.ndarray,
                                  confidence_percent: float = 0.95):
    logger.debug(f'Calculating confidence interval with confidence percent {confidence_percent}')
    return stats.t.interval(confidence_percent,
                            len(array) - 1,
                            loc=array.mean(),
                            scale=stats.sem(array))
