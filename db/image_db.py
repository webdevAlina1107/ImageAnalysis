import datetime
from typing import Optional

import numpy as np
import functools
import sqlite3
import os

SCRIPTS_STORAGE: str = os.path.join(os.path.dirname(__file__), 'scripts')
SCRIPT_EXTENSION: str = '.sql'


def get_sql_statement(script_name: str):
    script_location = os.path.join(SCRIPTS_STORAGE, script_name) + SCRIPT_EXTENSION
    if not os.path.isfile(script_location):
        pass
    with open(script_location, 'r') as script_file:
        return script_file.read()


def interacts_with_database(function):
    functools.wraps(function)

    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except sqlite3.DataError as error:
            pass
        except sqlite3.DatabaseError as error:
            pass
        except Exception as error:
            pass

    return wrapper


class ImageDatabase:
    def __init__(self):
        self.connection = sqlite3.connect('.db/datasource/images.db')
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

    @interacts_with_database
    def insert_image(self,
                     field_id: int,
                     image_bitmap: np.ndarray,
                     image_date: datetime.date):
        statement = get_sql_statement('insert_image')
        self.execute_statement(statement, field_id, image_bitmap, image_date)

    @interacts_with_database
    def insert_field(self,
                     field_id: int):
        statement = get_sql_statement('insert_field')
        self.execute_statement(statement, field_id)

    @interacts_with_database
    def insert_image_statistics(self,
                                image_id: int,
                                cloudiness: float,
                                index_average: float,
                                standard_deviation: float,
                                confidence_interval_lower: float,
                                confidence_interval_upper: float):
        statement = get_sql_statement('insert_image_statistics')
        self.execute_statement(statement,
                               image_id,
                               cloudiness,
                               index_average,
                               standard_deviation,
                               confidence_interval_lower,
                               confidence_interval_upper)

    @interacts_with_database
    def check_if_field_exists(self,
                              field_id: int):
        statement = get_sql_statement('check_if_field_exists')
        self.execute_statement(statement, field_id)
        return self.cursor.fetchone()

    @interacts_with_database
    def check_if_image_exist(self,
                             image_id: int):
        statement = get_sql_statement('check_if_image_exist')
        self.execute_statement(statement, image_id)
        return self.cursor.fetchone()

    @interacts_with_database
    def check_if_has_cached_statistics(self,
                                       image_id: int):
        statement = get_sql_statement('check_if_has_cached_statistics')
        self.execute_statement(statement, image_id)
        return self.cursor.fetchone()

    @interacts_with_database
    def select_images(self,
                      date_start: datetime.date = datetime.date.min,
                      date_end: datetime.date = datetime.date.max,
                      fetch_limit: Optional[int] = None):
        statement = 'SELECT * FROM image WHERE data between ? and ?'
        self.execute_statement(statement, date_start, date_end)
        return self.fetch_n(fetch_limit)

    @interacts_with_database
    def select_field_statistics(self,
                                field_id: int,
                                date_start: datetime.date = datetime.date.min,
                                date_end: datetime.date = datetime.date.max,
                                max_cloudiness: float = 1.0):
        statement = get_sql_statement('select_field_statistics')
        self.execute_statement(statement, field_id, date_start, date_end, max_cloudiness)
