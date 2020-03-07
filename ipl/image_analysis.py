import rasterio as rast
import numpy as np
from pathlib import Path
import scipy.stats as stats

IMAGE_DATA_TYPE = np.uint8
SPECIAL_VALUE: IMAGE_DATA_TYPE = IMAGE_DATA_TYPE(255)


def fill_cloud_bits_with_nan(array: np.ndarray) -> np.ndarray:
    non_clouds_range = (1, 253)
    masked_array = np.ma.masked_outside(array, *non_clouds_range)
    # Not np.nan because it's only for arrays with float data type only
    return masked_array.filled(SPECIAL_VALUE)


def read_image_bitmap(image_file_path: Path,
                      selected_band: int = 1
                      ) -> np.ndarray:
    with rast.open(image_file_path, 'r', dtype=IMAGE_DATA_TYPE) as raster:
        return raster.read(selected_band)


def construct_values_occurrences_map(array: np.ndarray):
    unique_values, counts = np.unique(array, return_counts=True)
    return {value: count for value, count in zip(unique_values, counts)}


def calculate_clouds_percentile(array: np.ndarray,
                                clouds_special_value: IMAGE_DATA_TYPE = SPECIAL_VALUE):
    return np.count_nonzero(array == clouds_special_value) / array.size


def calculate_confidence_interval(array: np.ndarray,
                                  confidence_percent: float = 0.95):
    return stats.t.interval(confidence_percent,
                            len(array) - 1,
                            loc=array.mean(),
                            scale=stats.sem(array))
