from enum import Enum
import datetime
import os

import rasterio as rast
import numpy as np
import re

from ipl.image_analysis import IMAGE_DATA_TYPE
from ipl._logging import logger
from ipl.errors import IPLError

IMAGE_FILE_NAME_PATTERN = re.compile(r"^(.+)_(.+)_.+_(.+)_.+$")


class SupportedDrivers(Enum):
    PNG = '.png'
    GTiff = '.tiff'
    GIF = '.gif'
    BMP = '.bmp'
    JPEG = '.jpg'

    @classmethod
    def drivers_list(cls):
        return list(cls.__members__.keys())


def read_image_bitmap(image_file_path: str) -> np.ndarray:
    logger.debug(f'Reading image at {image_file_path}, band = 1')
    with rast.open(image_file_path, 'r', dtype=IMAGE_DATA_TYPE) as raster:
        return raster.read(1)


def write_image_bitmap(image_file_path: str,
                       array: np.ndarray,
                       selected_driver: str = 'GTiff'):
    logger.debug(f'Writing image data to "{image_file_path}"')
    width, height = array.shape
    sharing_mode_on = selected_driver == 'GTiff'
    try:
        with rast.open(image_file_path, mode='w', driver=selected_driver,
                       width=width, height=height, count=1, dtype=IMAGE_DATA_TYPE,
                       sharing=sharing_mode_on) as image_file:
            image_file.write(array, 1)
    except rast.RasterioIOError as error:
        raise IPLError(f'Unable to export image, reason : {error}')


def parse_image_file_name(image_file_path: str):
    basename = os.path.splitext(os.path.basename(image_file_path))[0]
    match = re.fullmatch(IMAGE_FILE_NAME_PATTERN, basename)
    if match:
        timestamp = datetime.datetime.strptime(match.group(1), "%d%m%Y").date()
        field_id = match.group(2)  # I am not sure
        return field_id, timestamp
    else:
        return None


def import_locally_stored_image(image_file_path: str):
    try:
        file_meta_info = parse_image_file_name(image_file_path)
        if file_meta_info:
            timestamp, field_id = file_meta_info
            bitmap = read_image_bitmap(image_file_path)
            return field_id, bitmap, timestamp
        else:
            return None
    except Exception as error:
        raise IPLError(f'Unable to import image at "{image_file_path}", reason : "{error}"')


def import_images_folder(folder_path: str):
    try:
        directory_files = (os.path.join(folder_path, file)
                           for file in os.listdir(folder_path))
        directory_files = filter(os.path.isfile, directory_files)
        imported_images_data = []
        for file in directory_files:
            image_data = import_locally_stored_image(file)
            if image_data:
                imported_images_data.append(image_data)
    except Exception as error:
        raise IPLError(f'Unable to import folder at "{folder_path}", reason : "{error}"')
