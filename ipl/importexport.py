import datetime
import os

import rasterio as rast
import numpy as np
import re

from ipl.image_analysis import IMAGE_DATA_TYPE
from ipl._logging import logger
from ipl.errors import IPLError

IMAGE_FILE_NAME_PATTERN = re.compile(r"^(.+)_(.+)_.+_(.+)_.+$")


def read_image_bitmap(image_file_path: str,
                      selected_band: int = 1
                      ) -> np.ndarray:
    logger.debug(f'Reading image at {image_file_path}, band = {selected_band}')
    with rast.open(image_file_path, 'r', dtype=IMAGE_DATA_TYPE) as raster:
        return raster.read(selected_band)


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
