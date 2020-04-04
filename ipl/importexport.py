import contextlib
import datetime
import os
import re
import sys
from enum import Enum
from operator import attrgetter

import numpy as np
import rasterio as rast

from ipl.errors import IPLError
from ipl.image_analysis import IMAGE_DATA_TYPE
from ipl.logging_ import logger

IMAGE_FILE_NAME_PATTERN = re.compile(r"^(\d+)_(\d+)r(\d+)_NDVI_P(\d+)_(.+)$")


class SupportedDrivers(Enum):
    PNG = 'png'
    GTiff = 'tiff'
    GIF = 'gif'
    BMP = 'bmp'
    JPEG = 'jpg'

    @classmethod
    def drivers_list(cls):
        return list(cls.__members__.keys())

    @classmethod
    def formats_list(cls):
        value_getter = attrgetter('value')
        return list(map(value_getter, cls.__members__.values()))


def read_image_bitmap(image_file_path: str) -> np.ndarray:
    logger.debug(f'Reading image at {image_file_path}, band = 1')
    with rast.open(image_file_path) as raster:
        return raster.read(1).astype(IMAGE_DATA_TYPE)


@contextlib.contextmanager
def silence_logs():
    original_stderr = sys.stderr
    original_stdout = sys.stdout
    try:
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        sys.stderr = original_stderr
        sys.stdout = original_stdout


def write_image_bitmap(image_file_path: str,
                       array: np.ndarray,
                       selected_driver: str = 'GTiff'):
    logger.debug(f'Writing image data to "{image_file_path}"')
    width, height = array.shape
    sharing_mode_on = selected_driver == 'GTiff'
    if array.dtype is not np.uint8:
        array = array.astype(np.uint8)
    try:
        if os.path.isfile(image_file_path):
            os.remove(image_file_path)

        with silence_logs():
            with rast.open(image_file_path, mode='w', driver=selected_driver,
                           width=width, height=height, count=1, dtype=array.dtype,
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
        field_id = int(match.group(2))
        revision = int(match.group(3))
        mysterious_date = datetime.datetime.strptime(match.group(4), "%Y%m%d").date()
        satellite = match.group(5)
        return field_id, revision, capture_date, mysterious_date, satellite
    else:
        return None


def get_file_extension(file_path: str):
    return os.path.splitext(file_path)[1][1:]


def import_locally_stored_image(image_file_path: str):
    supported_extensions = SupportedDrivers.formats_list()
    try:
        file_extension = get_file_extension(image_file_path).lower()
        if file_extension in supported_extensions:
            file_meta_info = parse_image_file_name(image_file_path)
            if file_meta_info:
                field_id, revision, capture_date, mysterious_date, satellite = file_meta_info
                logger.debug(f'Importing image bit map for image at "{image_file_path}"')
                bitmap = read_image_bitmap(image_file_path)
                data_tuple = field_id, revision, bitmap, capture_date, satellite, mysterious_date
                return image_file_path, data_tuple
        else:
            logger.info('Unknown extension "%s" file at "%s"', file_extension, image_file_path)
    except Exception as error:
        raise IPLError(f'Unable to import image at "{image_file_path}", reason : {error}')

    return None


def import_images_folder(folder_path: str):
    try:
        directory_objects = list(os.path.join(folder_path, file) for file in os.listdir(folder_path))
        directory_files = list(filter(os.path.isfile, directory_objects))
        percentage_generator = ((index + 1) / len(directory_files) * 100
                                for index in range(0, len(directory_files)))
        for i, (percentage, file) in enumerate(zip(percentage_generator, directory_files)):
            image_data = import_locally_stored_image(file)
            if image_data:
                yield image_data
            logger.info('Processed %s files in %s folder, completed %.2f %%', i + 1, folder_path, percentage)

        sub_folders = filter(os.path.isdir, directory_objects)
        for sub_folder in sub_folders:
            yield from import_images_folder(sub_folder)
    except Exception as error:
        raise IPLError(f'Unable to import folder at "{folder_path}", reason : "{error}"')
