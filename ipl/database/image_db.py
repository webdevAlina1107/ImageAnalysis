import datetime
import functools
import io
import os
import sqlite3
from contextlib import contextmanager
from typing import List, Optional

import numpy as np
import pandas as pd

from ipl.errors import IPLError
from ipl.logging_ import logger

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
            message = f'Data error : {error}'
            raise IPLError(message)
        except sqlite3.DatabaseError as error:
            message = f'Database error @ {error}'
            raise IPLError(message)
        except IPLError:
            raise
        except Exception as error:
            message = f'Unexpected error @ {error}'
            raise IPLError(message)

    return wrapper


def _adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def _convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


class ImageDatabase:
    def __init__(self,
                 db_location: str = DATABASE_LOCATION):
        logger.debug('Registering adapters')
        sqlite3.register_adapter(np.ndarray, _adapt_array)
        sqlite3.register_converter("np_array", _convert_array)
        logger.debug(f'Connecting to database at "{db_location}"')
        self.connection = sqlite3.connect(db_location, detect_types=sqlite3.PARSE_DECLTYPES)
        logger.debug('Connected successfully !')
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
                     revision: int,
                     image_bitmap: np.ndarray,
                     capture_date: datetime.date,
                     capture_satellite: str,
                     mysterious_date: datetime.date):
        if not self.check_if_image_details_exist(field_id, revision,
                                                 capture_date, mysterious_date,
                                                 capture_satellite):
            if not self.check_if_field_exists(field_id):
                self.insert_field(field_id)
            statement = _get_sql_statement('insert_image')
            self.execute_statement(statement, field_id, revision, image_bitmap,
                                   capture_date, capture_satellite, mysterious_date)
            return self.cursor.lastrowid
        else:
            return None

    @_interacts_with_database
    def insert_field(self,
                     field_id: int):
        statement = _get_sql_statement('insert_field')
        with self.connection:
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
        with self.connection:
            self.execute_statement(statement,
                                   image_id,
                                   cloudiness,
                                   index_average,
                                   standard_deviation,
                                   confidence_interval_lower,
                                   confidence_interval_upper)
        return self.cursor.lastrowid

    @_interacts_with_database
    def check_if_image_details_exist(self,
                                     field_id: int,
                                     revision: int,
                                     capture_date: datetime.date,
                                     mysterious_date: datetime.date,
                                     capture_satellite: str):
        statement = _get_sql_statement('check_if_image_details_exist')
        self.execute_statement(statement, field_id, revision,
                               capture_date, mysterious_date, capture_satellite)
        return self.cursor.fetchone()[0]

    @_interacts_with_database
    def check_if_field_exists(self,
                              field_id: int):
        statement = _get_sql_statement('check_if_field_exists')
        self.execute_statement(statement, field_id)
        return self.cursor.fetchone()[0]

    @_interacts_with_database
    def check_if_image_exist(self,
                             image_id: int):
        statement = _get_sql_statement('check_if_image_exist')
        self.execute_statement(statement, image_id)
        return self.cursor.fetchone()[0]

    @_interacts_with_database
    def check_if_has_cached_statistics(self,
                                       image_id: int):
        statement = _get_sql_statement('check_if_has_cached_statistics')
        self.execute_statement(statement, image_id)
        return self.cursor.fetchone()[0]

    @_interacts_with_database
    def select_field_images(self,
                            field_id: int,
                            filtered_columns: Optional[List[str]] = None,
                            date_start: datetime.date = datetime.date.min,
                            date_end: datetime.date = datetime.date.max,
                            limit: Optional[int] = None):
        statement = _get_sql_statement('select_images')
        query_parameters = (field_id, date_start, date_end,)
        if limit is not None:
            statement += f' LIMIT {limit}'
        records = pd.read_sql_query(statement, self.connection, params=query_parameters)
        if filtered_columns:
            records = records.filter(items=filtered_columns)
        return records

    @_interacts_with_database
    def select_image(self,
                     image_id: int,
                     filtered_columns: Optional[List[str]] = None):
        statement = f'SELECT * FROM image WHERE image_id = ?'
        dataframe = pd.read_sql_query(statement, self.connection, params=(int(image_id),))
        if dataframe.shape and dataframe.shape[0] > 0:
            if filtered_columns:
                dataframe = dataframe.filter(filtered_columns)
            return dataframe.iloc[[0]]
        else:
            raise IPLError(f'Image with ID = {image_id} not found !')

    @_interacts_with_database
    def select_images_ids(self):
        statement = 'SELECT image_id from image'
        return pd.read_sql_query(statement, self.connection)['image_id']

    @_interacts_with_database
    def select_fields_ids(self):
        statement = 'SELECT field_id from field'
        return pd.read_sql_query(statement, self.connection)['field_id']

    @_interacts_with_database
    def select_field_statistics(self,
                                field_id: str,
                                filtered_columns: Optional[List[str]] = None,
                                date_start: datetime.date = datetime.date.min,
                                date_end: datetime.date = datetime.date.max,
                                max_cloudiness: float = 1.0):
        statement = _get_sql_statement('select_field_statistics')
        query_parameters = (field_id, date_start, date_end, max_cloudiness,)
        dataframe = pd.read_sql_query(statement, self.connection, params=query_parameters)
        if filtered_columns:
            return dataframe.filter(filtered_columns)
        return dataframe

    @contextmanager
    def disable_transactions(self):
        actual_level = self.connection.isolation_level
        try:
            self.connection.isolation_level = None
            yield
        finally:
            self.connection.isolation_level = actual_level

    @_interacts_with_database
    def make_vacuum(self):
        with self.disable_transactions():
            self.connection.execute('VACUUM')

    @_interacts_with_database
    def erase_all(self):
        erase_order = ['statistic_info', 'image', 'field']
        with self.connection:
            for table in erase_order:
                statement = f'DELETE FROM {table}'
                self.execute_statement(statement)
        self.make_vacuum()


class ImageDatabaseInstance:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = ImageDatabase(DATABASE_LOCATION)
        return cls.instance
