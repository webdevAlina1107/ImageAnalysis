import datetime
import os
import re
from enum import Enum

import numpy as np
import rasterio as rast

from ipl._logging import logger
from ipl.errors import IPLError
from ipl.image_analysis import IMAGE_DATA_TYPE

IMAGE_FILE_NAME_PATTERN = re.compile(r"^(.+)_(.+)_.+_(.+)_(.+)$")


class SupportedDrivers(Enum):
    PNG = 'png'
    GTiff = 'tiff'
    GIF = 'gif'
    BMP = 'bmp'
    JPEG = 'jpg'

    @classmethod
    def drivers_list(cls):
        return list(cls.__members__.keys())


def read_image_bitmap(image_file_path: str) -> np.ndarray:
    logger.debug(f'Reading image at {image_file_path}, band = 1')
    with rast.open(image_file_path) as raster:
        return raster.read(1).astype(IMAGE_DATA_TYPE)


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
    logger.debug(f'Parsing file meta info @ "{image_file_path}"')
    basename = os.path.splitext(os.path.basename(image_file_path))[0]
    match = re.fullmatch(IMAGE_FILE_NAME_PATTERN, basename)
    if match:
        capture_date = datetime.datetime.strptime(match.group(1), "%d%m%Y").date()
        field_id = match.group(2)  # I am not sure
        mysterious_date = datetime.datetime.strptime(match.group(3)[1:], "%Y%m%d").date()
        satellite = match.group(4)
        return field_id, capture_date, mysterious_date, satellite
    else:
        return None


def import_locally_stored_image(image_file_path: str):
    try:
        file_meta_info = parse_image_file_name(image_file_path)
        if file_meta_info:
            field_id, capture_date, mysterious_date, satellite = file_meta_info
            logger.debug(f'Importing image bit map for image at "{image_file_path}"')
            bitmap = read_image_bitmap(image_file_path)
            return field_id, bitmap, capture_date, satellite, mysterious_date
        else:
            return None
    except Exception as error:
        raise IPLError(f'Unable to import image at "{image_file_path}", reason : "{error}"')


def import_images_folder(folder_path: str):
    try:
        directory_files = (os.path.join(folder_path, file)
                           for file in os.listdir(folder_path))
        directory_files = list(filter(os.path.isfile, directory_files))
        imported_images_data = []
        for i, file in enumerate(directory_files):
            image_data = import_locally_stored_image(file)
            if image_data:
                imported_images_data.append(image_data)
            logger.info(f'Processed {i + 1} files out of {len(directory_files)} | '
                        f'Completed {((i + 1) / len(directory_files)) * 100} %')
        return imported_images_data
    except Exception as error:
        raise IPLError(f'Unable to import folder at "{folder_path}", reason : "{error}"')
