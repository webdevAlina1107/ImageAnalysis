import datetime
from typing import Optional

import numpy as np
import pandas as pd
import functools
import sqlite3
import os
import io

from ipl.errors import IPLError

FILE_LOCATION = os.path.abspath(os.path.dirname(__file__))
DATABASE_LOCATION = os.path.join(FILE_LOCATION, 'datasource', 'images.db')
SCRIPTS_STORAGE: str = os.path.join(FILE_LOCATION, 'scripts')
SCRIPT_EXTENSION: str = '.sql'


def _get_sql_statement(script_name: str):
    script_location = os.path.join(SCRIPTS_STORAGE, script_name) + SCRIPT_EXTENSION
    if not os.path.isfile(script_location):
        pass
    with open(script_location, 'r') as script_file:
        return script_file.read()


def _interacts_with_database(function):
    functools.wraps(function)

    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except sqlite3.DataError as error:
            message = f'DATA ERROR : "{error}"'
            raise IPLError(message)
        except sqlite3.DatabaseError as error:
            message = f'DATABASE ERROR : "{error}"'
            raise IPLError(message)
        except Exception as error:
            message = f'UNEXPECTED ERROR : "{error}"'
            raise IPLError(message)

    return wrapper


def _adapt_array(array: np.ndarray) -> memoryview:
    buffer = io.BytesIO()
    np.save(buffer, array)
    buffer.seek(0)
    return sqlite3.Binary(buffer.read())


def _convert_array(text: bytes) -> np.ndarray:
    buffer = io.BytesIO(text)
    buffer.seek(0)
    return np.load(buffer)


class ImageDatabase:
    def __init__(self,
                 db_location: str = DATABASE_LOCATION):
        sqlite3.register_adapter(np.ndarray, _adapt_array)
        sqlite3.register_converter("np_array", _convert_array)
        self.connection = sqlite3.connect(db_location)
        self.cursor = self.connection.cursor()

    def execute_statement(self,
                          statement: str,
                          *args):
        self.cursor.execute(statement, args)

    def fetch_n(self,
                fetch_limit: Optional[int]):
        if fetch_limit is None:
            return self.cursor.fetchall()
        else:
            return self.cursor.fetchmany(fetch_limit)

    @_interacts_with_database
    def insert_image(self,
                     field_id: int,
                     image_bitmap: np.ndarray,
                     image_date: datetime.date):
        statement = _get_sql_statement('insert_image')
        self.execute_statement(statement, field_id, image_bitmap, image_date)
        return self.cursor.lastrowid

    @_interacts_with_database
    def insert_field(self,
                     field_id: int):
        statement = _get_sql_statement('insert_field')
        self.execute_statement(statement, field_id)
        return field_id

    @_interacts_with_database
    def insert_image_statistics(self,
                                image_id: int,
                                cloudiness: float,
                                index_average: float,
                                standard_deviation: float,
                                confidence_interval_lower: float,
                                confidence_interval_upper: float):
        statement = _get_sql_statement('insert_image_statistics')
        self.execute_statement(statement,
                               image_id,
                               cloudiness,
                               index_average,
                               standard_deviation,
                               confidence_interval_lower,
                               confidence_interval_upper)
        return self.cursor.lastrowid

    @_interacts_with_database
    def check_if_field_exists(self,
                              field_id: int):
        statement = _get_sql_statement('check_if_field_exists')
        self.execute_statement(statement, field_id)
        return self.cursor.fetchone()

    @_interacts_with_database
    def check_if_image_exist(self,
                             image_id: int):
        statement = _get_sql_statement('check_if_image_exist')
        self.execute_statement(statement, image_id)
        return self.cursor.fetchone()

    @_interacts_with_database
    def check_if_has_cached_statistics(self,
                                       image_id: int):
        statement = _get_sql_statement('check_if_has_cached_statistics')
        self.execute_statement(statement, image_id)
        return self.cursor.fetchone()

    @_interacts_with_database
    def select_images(self,
                      field_id: int,
                      date_start: datetime.date = datetime.date.min,
                      date_end: datetime.date = datetime.date.max,
                      should_return_image_blob: bool = False):
        statement = _get_sql_statement('select_images')
        query_parameters = (field_id, date_start, date_end,)
        records = pd.read_sql_query(statement, self.connection, params=query_parameters)
        if not should_return_image_blob:
            del records['Image_bitmap']
        return records

    @_interacts_with_database
    def select_field_statistics(self,
                                field_id: int,
                                date_start: datetime.date = datetime.date.min,
                                date_end: datetime.date = datetime.date.max,
                                max_cloudiness: float = 1.0):
        statement = _get_sql_statement('select_field_statistics')
        query_parameters = (field_id, date_start, date_end, max_cloudiness,)
        return pd.read_sql_query(statement, self.connection, params=query_parameters)


class ImageDatabaseInstance:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = ImageDatabase(DATABASE_LOCATION)
        return cls.instance
