import logging
from typing import Optional

logger = logging.getLogger('IPL')


def configure_logger(is_debug: bool,
                     logs_file: Optional[str] = None):
    console_format = '%(message)s'
    formatter = logging.Formatter(console_format)
    console_handler = logging.StreamHandler()
    logging_level = logging.DEBUG if is_debug else logging.INFO
    console_handler.setLevel(logging_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logs_file:
        try:
            file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            date_format = '%Y-%m-%d %H:%M:%S'
            file_formatter = logging.Formatter(file_format, date_format)
            file_handler = logging.FileHandler(logs_file, 'w')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
        except Exception as error:
            logger.warning(f'Unable to create log file, reason : "{error}"')
