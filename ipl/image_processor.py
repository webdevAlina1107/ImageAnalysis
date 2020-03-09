import argparse
import datetime
import os
from pathlib import Path
from pprint import pformat
from typing import List, Optional
import pandas as pd
from tabulate import tabulate
import traceback

from ipl._logging import configure_logger, logger
from ipl.image_analysis import calculate_all_statistics
from ipl.errors import IPLError
import ipl.db.image_db as image_db
import ipl.importexport as io


def _parse_date(string: str):
    pattern = '%d/%m/%Y'
    try:
        return datetime.datetime.strptime(string, pattern).date()
    except ValueError as error:
        msg = f'Not a valid date: "{string}", reason: "{error}"'
        raise argparse.ArgumentTypeError(msg)


def _parse_file_path(path: str):
    if not os.path.isfile(path):
        message = f'Provided string is not a valid file path, "{path}"'
        raise argparse.ArgumentTypeError(message)
    return path


def _parse_float_in_range(string: str):
    try:
        value = float(string)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{string} not a floating-point literal")
    if 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"{value} not in range [0.0, 1.0]")

    return value


def _collect_images(field_ids: List[int],
                    start_date: datetime.date,
                    end_date: datetime.date,
                    should_sort_images: bool = True,
                    should_return_image_bitmap: bool = False):
    database = image_db.ImageDatabaseInstance()
    select_images = database.select_images
    dataframes_list: List[pd.DataFrame] = []
    for field_id in field_ids:
        field_images = select_images(field_id=field_id,
                                     date_start=start_date,
                                     date_end=end_date,
                                     should_return_image_blob=should_return_image_bitmap)
        dataframes_list.append(field_images)

    return pd.concat(dataframes_list, sort=should_sort_images)


def database_view(id_: List[int],
                  head: Optional[int],
                  start: datetime.date,
                  end: datetime.date,
                  **kwargs):
    dataframe = _collect_images(field_ids=id_,
                                start_date=start,
                                end_date=end,
                                should_return_image_bitmap=False)
    shown_dataframe = dataframe.head(head) if head else dataframe
    print(tabulate(shown_dataframe, headers='keys', tablefmt='psql'))


def visualize_clouds(id_: int,
                     start: datetime.date,
                     end: datetime.date,
                     **kwargs):
    print('DONE')


def visualize_occurrences(file: Optional[Path],
                          id_: Optional[int],
                          **kwargs):
    print('DONE')


def visualize_statistics(id_: List[int],
                         start: datetime.date,
                         end: datetime.date,
                         cloudiness: float,
                         **kwargs):
    print('DONE')


def import_images(import_location: str,
                  cache: bool,
                  **kwargs):
    if os.path.isdir(import_location):
        inserted_images_array = io.import_images_folder(import_location)
    else:
        inserted_images_array = [io.import_locally_stored_image(import_location)]
    database = image_db.ImageDatabaseInstance()
    inserted_images_ids = [database.insert_image(*image) for image in inserted_images_array]
    if cache:
        bitmaps_generator = (image[1] for image in inserted_images_array)
        for image_id, bitmap in zip(inserted_images_ids, bitmaps_generator):
            statistics = calculate_all_statistics(bitmap)
            database.insert_image_statistics(image_id, *statistics)


def export_images(export_location: Path,
                  start: datetime.date,
                  end: datetime.date,
                  all_: bool,
                  id_: List[int],
                  **kwargs):
    if not os.path.isdir(export_location):
        raise IPLError('Unable to export data to non-existent directory')
    database = image_db.ImageDatabaseInstance()
    if all_:
        id_ = database.select_fields_ids()
    dataframe = _collect_images(field_ids=id_,
                                start_date=start,
                                end_date=end,
                                should_return_image_bitmap=True)
    for index, row in dataframe.iterrows():
        pass


def process_images(file: Path,
                   id_: List[int],
                   cache: bool,
                   **kwargs):
    print('DONE')


def cmdline_arguments():
    parser = argparse.ArgumentParser(description='Utility to process images')
    parser.add_argument('--debug', action='store_true',
                        help='Toggles debug info printout')
    parser.add_argument('--log_file', dest='log_file', default=None,
                        type=str, metavar='PATH/TO/LOGS/FILE',
                        help='Path to file where to store logs')
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    # IMPORT SUBPARSER
    import_parser = subparsers.add_parser('import', help='Imports images into database')
    import_parser.add_argument('import_location', type=str, nargs='+',
                               metavar='PATH/TO/DATASET',
                               help='Location of file or folder to import,'
                                    ' can be specified multiple times')
    import_parser.add_argument('-cc', '--calculations_cache', dest='cache', action='store_true',
                               help='Enables calculations caching while importing image')
    import_parser.set_defaults(function=import_images)

    # EXPORT SUBPARSER

    export_parser = subparsers.add_parser('export', help='Exports images out of database')
    export_parser.add_argument('export_location', type=str, metavar='PATH/TO/EXPORT/FOLDER',
                               help='Location of folder to export selected data')
    export_parser.add_argument('--start_date', dest='start', type=_parse_date, default=datetime.date.min,
                               help='Start date of a timeline', metavar='DD/MM/YYYY')
    export_parser.add_argument('--end_date', dest='end', type=_parse_date, default=datetime.date.max,
                               help='End date of a timeline', metavar='DD/MM/YYYY')
    selection_type = export_parser.add_mutually_exclusive_group(required=True)
    selection_type.add_argument('--id', dest='id_', default=None, type=int, nargs='+',
                                metavar='FIELD_ID', help='ID of a field in a database')
    selection_type.add_argument('--all', action='store_true', dest='all_',
                                help='Exports without filtering by id')
    export_parser.set_defaults(function=export_images)

    # PROCESSING SUBPARSER

    processing_parser = subparsers.add_parser('process', help='Process an image in a database or file')
    processing_parser.add_argument('--file', dest='file', default=None, type=_parse_file_path,
                                   help='Path to image file', metavar='PATH/TO/IMAGE')
    processing_parser.add_argument('--id', dest='id_', default=None, type=int, nargs='+',
                                   metavar='IMAGE_ID', help='IDs of processed images')
    processing_parser.add_argument('-cc', '--calculations_cache', dest='cache', action='store_true',
                                   help='Enables calculations caching')
    processing_parser.set_defaults(function=process_images)

    # DB VIEW SUBPARSER

    db_view_parser = subparsers.add_parser('view', help='View DB records')

    image_selection_group_ = db_view_parser.add_mutually_exclusive_group(required=True)
    image_selection_group_.add_argument('--id', nargs='+', type=int, dest='id_',
                                        metavar='FIELD_ID', help="IDs of viewed fields' images")
    image_selection_group_.add_argument('--all', action='store_true', dest='all_',
                                        help='Selects all records')
    db_view_parser.add_argument('--head', type=int, dest='head', default=None,
                                help='Max amount of printed records')
    db_view_parser.add_argument('--start_date', dest='start', type=_parse_date, default=datetime.date.min,
                                help='Start date of a timeline', metavar='DD/MM/YYYY')
    db_view_parser.add_argument('--end_date', dest='end', type=_parse_date, default=datetime.date.max,
                                help='End date of a timeline', metavar='DD/MM/YYYY')
    db_view_parser.set_defaults(function=database_view)

    # VISUALIZATION SUBPARSERS

    visualization_parser = subparsers.add_parser('visualize', help='Visualize statistical data')
    visualization_subparsers = visualization_parser.add_subparsers(dest='diagram')
    visualization_subparsers.required = True

    # CLOUDS INFO VISUALIZATION SUBPARSER

    clouds_parser = visualization_subparsers.add_parser('clouds',
                                                        help='Histogram to visualize cloudiness '
                                                             'of images on a timeline')
    clouds_parser.add_argument('--id', required=True, dest='id_', type=int,
                               metavar='FIELD_ID', help='ID of a processed field')
    clouds_parser.add_argument('--start_date', dest='start', default=datetime.date.min, type=_parse_date,
                               help='Start of analysed timeline', metavar='DD/MM/YYYY')
    clouds_parser.add_argument('--end_date', dest='end', default=datetime.date.max, type=_parse_date,
                               help='End of analysed timeline', metavar='DD/MM/YYYY')
    clouds_parser.set_defaults(function=visualize_clouds)

    # VALUES OCCURRENCES VISUALIZATION SUBPARSER

    occurrences_parser = visualization_subparsers.add_parser('occurrences',
                                                             help='Histogram to visualize frequency'
                                                                  ' of values occurrences in an image')
    image_selection_group = occurrences_parser.add_mutually_exclusive_group(required=True)
    image_selection_group.add_argument('--file', dest='file', default=None, type=_parse_file_path,
                                       help='Path to image file', metavar='PATH/TO/IMAGE')
    image_selection_group.add_argument('--id', dest='id_', default=None, type=int,
                                       metavar='IMAGE_ID', help='ID of processed image')
    occurrences_parser.set_defaults(function=visualize_occurrences)

    # STATISTICS VISUALIZATION SUBPARSER

    statistics_parser = visualization_subparsers.add_parser('statistics',
                                                            help='Diagram to visualize multiple'
                                                                 ' datasets statistical data')
    statistics_parser.add_argument('--id', nargs='+', required=True, type=int, dest='id_',
                                   metavar='FIELD_ID', help='IDs of visualized fields')
    statistics_parser.add_argument('--start_date', dest='start', default=datetime.date.min, type=_parse_date,
                                   help='Start of analysed timeline', metavar='DD/MM/YYYY')
    statistics_parser.add_argument('--end_date', dest='end', default=datetime.date.max, type=_parse_date,
                                   help='End of analysed timeline', metavar='DD/MM/YYYY')
    statistics_parser.add_argument('--max_cloudiness', dest='cloudiness', default=0.5, type=_parse_float_in_range,
                                   metavar='[0.0, 1.0]', help='Filtering cloudiness percent')
    statistics_parser.set_defaults(function=visualize_statistics)

    arguments = parser.parse_args()

    if arguments.command == 'process':
        if arguments.file and (arguments.id or arguments.cache):
            parser.error('Unable to parse mutually exclusive group ["file"] and ["id", "calculations_cache"]')

    return arguments


def main():
    arguments = cmdline_arguments()
    configure_logger(is_debug=arguments.debug,
                     logs_file=arguments.log_file)
    try:
        function = arguments.function
        logger.debug(f'Starting target function with arguments : \n{pformat(vars(arguments), indent=4)}')
        function(**vars(arguments))
        logger.debug('Action succeeded !')
    except IPLError as error:
        logger.critical(f'Database error : "{error}"')
        logger.debug(traceback.format_exc())
    except Exception as error:
        logger.critical(f'Something went wrong : "{error}"')
        logger.debug(traceback.format_exc())


if __name__ == '__main__':
    main()
