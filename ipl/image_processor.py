import argparse
import datetime
import os
from pprint import pformat
from typing import List, Optional, Iterable
from itertools import tee, islice
import pandas as pd
from tabulate import tabulate
import numpy as np
import calendar
import traceback

from ipl._logging import configure_logger, logger
import ipl.image_analysis as image_anal
from ipl.errors import IPLError
import ipl.database.image_db as image_db
import ipl.importexport as importexport
import ipl.visualization as visualization


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


def _add_months(initial_date: datetime.date,
                months: int):
    month = initial_date.month - 1 + months
    year = initial_date.year + month
    month = month % 12 + 1
    month_range = calendar.monthrange(year, month)[1]
    day = min(initial_date.day, month_range)
    return datetime.date(year, month, day)


def _collect_cloudiness_data(cloudiness_data: Iterable[float],
                             epsilon: float = 0.01):
    """Returns non-clouded, partially clouded and fully clouded count"""

    def generate_estimation_tuple(cloudiness):
        return (cloudiness < epsilon,
                epsilon < cloudiness < 1 - epsilon,
                cloudiness > 1 - epsilon)

    non_gen, partially_gen, fully_gen = tee(generate_estimation_tuple(cloudiness)
                                            for cloudiness in cloudiness_data)
    return (sum(1 for non, partially, fully in non_gen if non),
            sum(1 for non, partially, fully in non_gen if partially),
            sum(1 for non, partially, fully in non_gen if fully))


def _calculate_images_statistics(images_array: List[np.ndarray],
                                 index_column: Optional[np.ndarray] = None):
    cloud_rate = np.empty(len(images_array), np.double)
    average = np.empty(len(images_array), np.double)
    std = np.empty(len(images_array), np.double)
    ci_lower = np.empty(len(images_array), np.double)
    ci_upper = np.empty(len(images_array), np.double)

    for index_, image in enumerate(images_array):
        average[index_] = np.mean(image)
        std[index_] = np.std(image)
        ci_lower[index_], ci_upper[index_] = image_anal.calculate_confidence_interval(image)
        image = image_anal.fill_cloud_bits_with_value(image)
        cloud_rate[index_] = image_anal.calculate_clouds_percentile(image)

    data_sources = (cloud_rate, average, std, ci_lower, ci_upper)
    columns = ['cloud_rate', 'ndvi_average', 'standard deviation', 'ci_lower', 'ci_upper']
    df_initializer = {label: source for label, source in zip(columns, data_sources)}
    dataframe = pd.DataFrame(df_initializer)
    if index_column:
        dataframe['Image ID'] = index_column
        dataframe.set_index('Image ID')
    return dataframe


def _collect_images(field_ids: List[int],
                    start_date: datetime.date,
                    end_date: datetime.date,
                    filtered_columns: Optional[List[str]] = None,
                    should_sort_images: bool = True):
    database = image_db.ImageDatabaseInstance()

    select_images = database.select_field_images
    dataframes_list: List[pd.DataFrame] = []
    for field_id in field_ids:
        field_images = select_images(field_id=field_id,
                                     date_start=start_date,
                                     date_end=end_date,
                                     filtered_columns=filtered_columns)
        dataframes_list.append(field_images)

    if dataframes_list:
        return pd.concat(dataframes_list, sort=should_sort_images)
    else:
        return None


def database_view(id_: Optional[List[int]],
                  head: Optional[int],
                  start: datetime.date,
                  end: datetime.date,
                  **kwargs):
    required_columns = ['field_id', 'image_id', 'capture_date', 'mysterious_date', 'satellite']
    database = image_db.ImageDatabaseInstance()
    if not id_:
        id_ = database.select_fields_ids()
    dataframe = _collect_images(field_ids=id_,
                                start_date=start,
                                end_date=end,
                                filtered_columns=required_columns)
    if dataframe:
        shown_dataframe = dataframe.head(head) if head else dataframe
        labels = [label.replace('_', ' ').capitalize() for label in required_columns]
        print(tabulate(shown_dataframe, headers=labels, tablefmt='psql'))
    else:
        print("No suitable records found !")


def visualize_clouds(id_: int,
                     start: datetime.date,
                     end: datetime.date,
                     **kwargs):
    database = image_db.ImageDatabaseInstance()
    required_columns = ['image_id', 'cloud_rate', 'capture_date']
    cached_statistics = database.select_field_statistics(id_,
                                                         filtered_columns=required_columns,
                                                         date_start=start,
                                                         date_end=end)
    calculated_images_set = set(cached_statistics['image_id'])
    cloud_rates = list(cached_statistics['cloud_rate'])
    capture_dates = list(cached_statistics['capture_date'])
    del cached_statistics

    required_columns = ['image_id', 'image_data', 'capture_date']
    all_images = database.select_field_images(field_id=id_,
                                              filtered_columns=required_columns,
                                              date_start=start,
                                              date_end=end)
    for index, row in all_images.iterrows():
        image_id = row['image_id']
        if image_id not in calculated_images_set:
            image_bitmap = row['image_data']
            image_bitmap = image_anal.fill_cloud_bits_with_value(image_bitmap)
            cloud_rate = image_anal.calculate_clouds_percentile(image_bitmap)
            cloud_rates.append(cloud_rate)
            capture_dates.append(row['capture_date'])
    assert len(cloud_rates) == len(capture_dates)
    if len(capture_dates) > 0:
        minimal_goal_date = _add_months(capture_dates[0], 1)
        dates_list = []
        statistics_arrays = [[], [], []]
        index_start, index_end = None, 0

        def _update_statistics(new_date: datetime.date):
            dates_list.append(new_date)
            cloud_rates_generator = islice(cloud_rates, index_start, index_end)
            for index_, statistics_item in enumerate(_collect_cloudiness_data(cloud_rates_generator)):
                statistics_arrays[index_].append(statistics_item)

        for index, date in enumerate(capture_dates):
            if date > dates_list[-1]:
                index_start, index_end = index_end, index
                _update_statistics(date)
                minimal_goal_date = _add_months(date, 1)
        index_start, index_end = index_end, len(capture_dates)
        _update_statistics(minimal_goal_date)
        visualization.plot_clouds_impact_for_a_period(dates_list, *statistics_arrays)
        visualization.show_plots()
    else:
        logger.debug('Unable to start visualization, no data found')


def visualize_occurrences(file: Optional[str],
                          id_: Optional[int],
                          **kwargs):
    if file:
        bitmap = importexport.read_image_bitmap(file)
    else:
        database = image_db.ImageDatabaseInstance()
        bitmap = database.select_image(id_)['image_data']

    unique_value_occurs = image_anal.construct_values_occurrences_map(bitmap)
    visualization.plot_values_frequencies(unique_value_occurs)
    visualization.show_plots()


def visualize_statistics(id_: List[int],
                         start: datetime.date,
                         end: datetime.date,
                         max_cloudiness: float,
                         **kwargs):
    database = image_db.ImageDatabaseInstance()
    required_fields = ['image_id', 'capture_date', 'index_weighted_avg',
                       'confidence_interval_lower', 'confidence_interval_upper']
    have_plotted_anything = False
    for field_id in id_:
        cached_statistics = database.select_field_statistics(field_id=field_id,
                                                             filtered_columns=required_fields,
                                                             date_start=start,
                                                             date_end=end,
                                                             max_cloudiness=max_cloudiness)
        cached_images_ids = set(cached_statistics['image_id'])
        required_fields = ['image_id', 'image_data', 'capture_date']
        other_images = database.select_field_images(field_id=field_id,
                                                    filtered_columns=required_fields,
                                                    date_start=start,
                                                    date_end=end)
        for index, image in other_images.iterrows():
            image_id = image['image_id']
            if image_id not in cached_images_ids:
                bitmap: np.ndarray = image['image_data']
                bitmap = image_anal.fill_cloud_bits_with_value(bitmap)
                cloud_rate = image_anal.calculate_clouds_percentile(bitmap)
                if cloud_rate < max_cloudiness:
                    capture_date = image['capture_date']
                    mean = np.nanmean(bitmap)
                    lower_ci, upper_ci = image_anal.calculate_confidence_interval(bitmap)
                    series = pd.Series([image_id, capture_date, mean, lower_ci, upper_ci],
                                       index=cached_statistics.columns)
                    cached_statistics.append(series, ignore_index=True)

        if cached_statistics.shape and cached_statistics.shape[0]:
            visualization.plot_statistics_for_a_period(time_stamps=cached_statistics['capture_date'],
                                                       mean=cached_statistics['index_weighted_avg'],
                                                       lower_ci=cached_statistics['confidence_interval_lower'],
                                                       upper_ci=cached_statistics['confidence_interval_upper'],
                                                       legend_name=str(field_id))
            have_plotted_anything = True

    if have_plotted_anything:
        visualization.show_plots()
    else:
        logger.debug('Unable to start visualization, no data found')


def import_images(import_location: List[str],
                  cache: bool,
                  **kwargs):
    for location in import_location:
        if os.path.isdir(location):
            inserted_images_array = importexport.import_images_folder(location)
        else:
            inserted_images_array = [importexport.import_locally_stored_image(location)]
        database = image_db.ImageDatabaseInstance()
        for field_id in (image_data[0] for image_data in inserted_images_array):
            if not database.check_if_field_exists(field_id):
                database.insert_field(field_id)
        inserted_images_ids = [database.insert_image(*image) for image in inserted_images_array]
        if cache:
            bitmaps_generator = (image[1] for image in inserted_images_array)
            for image_id, bitmap in zip(inserted_images_ids, bitmaps_generator):
                statistics = image_anal.calculate_all_statistics(bitmap)
                database.insert_image_statistics(image_id, *statistics)


def export_images(export_location: str,
                  start: datetime.date,
                  end: datetime.date,
                  driver: str,
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
                                end_date=end)

    selected_extension = importexport.SupportedDrivers[driver].value

    for index, row in dataframe.iterrows():
        capture_date = row['capture_date']
        satellite = row['satellite']
        mysterious_date = row['mysterious_date']
        field_id = row['field_id']
        bitmap = row['image_data']
        file_name = f'{capture_date}_{field_id}_{mysterious_date}_{satellite}.{selected_extension}'
        file_path = os.path.join(export_location, file_name)
        importexport.write_image_bitmap(file_path, bitmap, driver)


def process_images(file: Optional[str],
                   id_: Optional[List[int]],
                   all_: bool,
                   cache: bool,
                   **kwargs):
    if file:
        processed_images = [importexport.read_image_bitmap(file)]
    else:
        database = image_db.ImageDatabaseInstance()
        if all_:
            id_ = database.select_images_ids()
        required_fields = ['image_data', 'image_id']
        images_data = [database.select_image(image_id, required_fields) for image_id in id_]
        id_ = [data['image_id'] for data in images_data]
        processed_images = [data['image_data'][0] for data in images_data]

    dataframe = _calculate_images_statistics(processed_images, id_)
    labels = [label.replace('_', ' ').capitalize() for label in dataframe.columns.values]
    print(tabulate(dataframe, headers=labels, tablefmt='psql'))
    if cache and id_:
        database = image_db.ImageDatabaseInstance()
        for image_id, (index, row) in zip(id_, dataframe.iterrows()):
            if not database.check_if_has_cached_statistics(image_id):
                cloud_rate = row['cloud_rate']
                ndvi_average = row['ndvi_average']
                std = row['standard_deviation']
                lower_ci = row['lower_ci']
                upper_ci = row['upper_ci']
                database.insert_image_statistics(image_id, cloud_rate, ndvi_average,
                                                 std, lower_ci, upper_ci)


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
    drivers_list = importexport.SupportedDrivers.drivers_list()

    export_parser = subparsers.add_parser('export', help='Exports images out of database')
    export_parser.add_argument('export_location', type=str, metavar='PATH/TO/EXPORT/FOLDER',
                               help='Location of folder to export selected data')
    export_parser.add_argument('--driver', type=str, choices=drivers_list,
                               dest='driver', default='GTiff',
                               help='Driver for image exporting')
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
    db_images_selection = processing_parser.add_mutually_exclusive_group()
    db_images_selection.add_argument('--id', dest='id_', default=None, type=int, nargs='+',
                                     metavar='IMAGE_ID', help='IDs of processed images')
    db_images_selection.add_argument('--all', dest='all_', action='store_true',
                                     help='Processes all images in database')
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
    statistics_parser.add_argument('--max_cloudiness', dest='max_cloudiness', default=0.5, type=_parse_float_in_range,
                                   metavar='[0.0, 1.0]', help='Filtering cloudiness percent')
    statistics_parser.set_defaults(function=visualize_statistics)

    arguments = parser.parse_args()

    if arguments.command == 'process':
        if arguments.file and (arguments.id_ or arguments.all_) or (not arguments.file and not arguments.id_
                                                                    and not arguments.all_):
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
        logger.critical(f'Database error @ {error}')
        logger.debug(traceback.format_exc())
    except Exception as error:
        logger.critical(f'Something went wrong @ {error}')
        logger.debug(traceback.format_exc())


if __name__ == '__main__':
    main()
